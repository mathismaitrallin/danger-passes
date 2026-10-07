"""
Modèle de dangerosité : probabilité qu'une passe soit suivie d'un tir de la même équipe
(même possession, dans les 15 secondes).

Validation croisée groupée par match (GroupKFold) : chaque passe est notée par un modèle
qui n'a jamais vu son match. Les scores affichés dans l'application sont ces prédictions
"hors échantillon".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CIBLE = "tir_15s"

# Variables utilisées par le modèle, regroupées par famille (pour l'importance et l'affichage)
FAMILLES = {
    "Distance au but (arrivée)": ["dist_but_fin", "x_fin"],
    "Angle de tir (arrivée)": ["angle_but_fin", "ecart_axe_fin", "dans_surface_fin"],
    "Point de départ": ["x_debut", "dist_but_debut"],
    "Progression vers le but": ["progression", "direction_but", "longueur"],
    "Adversaires éliminés": ["adv_elimines"],
    "Défenseurs restants": ["defenseurs_restants", "adv_surface"],
    "Passe dans le dos de la défense": ["derriere_defense"],
    "Cône de tir obstrué": ["adv_cone_tir", "gardien_dans_cone"],
    "Pression sur le receveur": ["dist_adv_fin", "adv_5m_fin"],
    "Pression sur le passeur": ["dist_adv_debut", "passeur_sous_pression"],
    "Soutien offensif": ["coequipiers_devant", "coequipiers_surface"],
    "Visibilité 360": ["adv_visibles"],
    "Technique de passe": ["en_profondeur", "centre", "renversement", "passe_en_retrait", "hauteur_*"],
    "Coup de pied arrêté": ["type_passe_*"],
    "Contexte de jeu": ["phase_jeu_*"],
}

CATEGORIELLES = ["type_passe", "hauteur", "phase_jeu"]

# Variables de position uniquement (sans le contexte 360) : sert de référence
POSITION = ["x_debut", "dist_but_debut", "x_fin", "dist_but_fin", "angle_but_fin", "ecart_axe_fin",
            "dans_surface_fin"]

DERNIER_TIERS = 70.0  # x (m) à partir duquel une passe arrive dans le dernier tiers

# Situation du passeur avant la passe : tout ce qu'on sait sans connaître la destination.
# Sert au modèle de "valeur avant la passe" (valeur ajoutée = danger après - valeur avant).
SITUATION_AVANT = ["x_debut", "dist_but_debut", "angle_but_debut", "ecart_axe_debut", "dans_surface_debut",
                   "defenseurs_restants_debut", "dist_adv_debut", "adv_5m_debut", "passeur_sous_pression",
                   "coequipiers_devant_debut", "adv_visibles"]
CATEGORIELLES_AVANT = ["type_passe", "phase_jeu"]

ZONES = {"Propre moitié": (0, 52.5), "Milieu offensif": (52.5, 70), "Dernier tiers": (70, 88.5),
         "Surface et abords": (88.5, 106)}


def matrice(passes: pd.DataFrame) -> pd.DataFrame:
    """Matrice de variables numériques (catégories encodées en indicatrices)."""
    colonnes = [c for cols in FAMILLES.values() for c in cols if not c.endswith("*")]
    X = passes[colonnes].astype(float)
    indicatrices = pd.get_dummies(passes[CATEGORIELLES], prefix=CATEGORIELLES, dtype=float)
    return pd.concat([X, indicatrices], axis=1)


def colonnes_famille(X: pd.DataFrame, famille: str) -> list[str]:
    cols = []
    for c in FAMILLES[famille]:
        cols += [x for x in X.columns if x.startswith(c[:-1])] if c.endswith("*") else [c]
    return cols


def ancienne_formule(passes: pd.DataFrame) -> pd.Series:
    """
    Score de la première version du projet (poids choisis à la main, normalisation min-max),
    recalculé sur ces données pour servir de point de comparaison.
    """
    vitesse = passes["longueur"] / passes["duree"].clip(lower=0.01)
    norm = lambda s: (s - s.min()) / (s.max() - s.min())
    return (0.25 * norm(passes["progression"])
            + 0.20 * norm(passes["direction_but"])
            + 0.15 * (1 - norm(passes["dist_but_fin"]))
            + 0.15 * norm(passes["adv_elimines"])
            + 0.10 * (1 - norm(passes["dist_adv_fin"]))
            + 0.10 * norm(vitesse.fillna(vitesse.median()))
            + 0.05 * (1 - norm(passes["adv_5m_fin"])))


def _boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=80,
        l2_regularization=1.0, random_state=0,
    )


def modeles() -> dict:
    return {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
        "Gradient boosting": _boosting(),
    }


def _predictions_hors_echantillon(modele, X: pd.DataFrame, y: np.ndarray, groupes: np.ndarray,
                                  n_plis: int) -> np.ndarray:
    oof = np.zeros(len(X))
    for apprentissage, test in GroupKFold(n_plis).split(X, y, groupes):
        modele.fit(X.iloc[apprentissage], y[apprentissage])
        oof[test] = modele.predict_proba(X.iloc[test])[:, 1]
    return oof


def _metriques(y: np.ndarray, score: np.ndarray, proba: bool) -> dict:
    m = {"auc": round(float(roc_auc_score(y, score)), 4)}
    if proba:
        base = np.full_like(score, y.mean(), dtype=float)
        brier, brier_base = brier_score_loss(y, score), brier_score_loss(y, base)
        m.update({
            "brier": round(float(brier), 5),
            "brier_skill": round(float(1 - brier / brier_base), 4),
            "log_loss": round(float(log_loss(y, score)), 5),
        })
    return m


def bootstrap_gain_auc(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, groupes: np.ndarray,
                       n: int = 300) -> dict:
    """Intervalle de confiance à 95 % du gain d'AUC (a - b), en ré-échantillonnant des matchs entiers."""
    rng = np.random.default_rng(0)
    matchs = np.unique(groupes)
    index = {m: np.flatnonzero(groupes == m) for m in matchs}
    gains = []
    for _ in range(n):
        tirage = np.concatenate([index[m] for m in rng.choice(matchs, len(matchs), replace=True)])
        gains.append(roc_auc_score(y[tirage], score_a[tirage]) - roc_auc_score(y[tirage], score_b[tirage]))
    bas, haut = np.percentile(gains, [2.5, 97.5])
    return {"gain": round(float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b)), 4),
            "ic95": [round(float(bas), 4), round(float(haut), 4)]}


def importance_par_famille(modele, X: pd.DataFrame, y: np.ndarray, groupes: np.ndarray,
                           n_plis: int = 5, repetitions: int = 3) -> dict[str, float]:
    """Perte d'AUC quand on mélange toutes les colonnes d'une famille (importance par permutation)."""
    rng = np.random.default_rng(0)
    pertes = {f: [] for f in FAMILLES}
    for apprentissage, test in GroupKFold(n_plis).split(X, y, groupes):
        modele.fit(X.iloc[apprentissage], y[apprentissage])
        X_test, y_test = X.iloc[test], y[test]
        auc_ref = roc_auc_score(y_test, modele.predict_proba(X_test)[:, 1])
        for famille in FAMILLES:
            cols = colonnes_famille(X, famille)
            for _ in range(repetitions):
                X_perm = X_test.copy()
                ordre = rng.permutation(len(X_perm))
                X_perm[cols] = X_perm[cols].to_numpy()[ordre]
                pertes[famille].append(auc_ref - roc_auc_score(y_test, modele.predict_proba(X_perm)[:, 1]))
    return {f: round(float(np.mean(v)), 4) for f, v in sorted(pertes.items(), key=lambda kv: -np.mean(kv[1]))}


def entrainer_et_evaluer(passes: pd.DataFrame, n_plis: int = 5) -> tuple[pd.Series, dict]:
    """
    Compare les approches et renvoie (probabilités hors échantillon du meilleur modèle, rapport).
    """
    X = matrice(passes)
    y = passes[CIBLE].to_numpy().astype(int)
    groupes = passes["match_id"].to_numpy()

    scores = {
        "Ancienne formule (poids à la main)": (ancienne_formule(passes).to_numpy(), False),
        "Distance au but seule": (-passes["dist_but_fin"].to_numpy(), False),
        "Position seule (sans 360)": (
            _predictions_hors_echantillon(_boosting(), X[POSITION], y, groupes, n_plis), True),
    }
    for nom, modele in modeles().items():
        scores[nom] = (_predictions_hors_echantillon(modele, X, y, groupes, n_plis), True)

    dernier_tiers = passes["x_fin"].to_numpy() >= DERNIER_TIERS
    resultats = {nom: _metriques(y, s, p) for nom, (s, p) in scores.items()}
    resultats_dernier_tiers = {nom: _metriques(y[dernier_tiers], s[dernier_tiers], p)
                               for nom, (s, p) in scores.items()}

    # on affiche des probabilités : le critère est la log loss (qualité + calibration)
    meilleur = min(modeles(), key=lambda n: resultats[n]["log_loss"])
    proba = scores[meilleur][0]

    # Calibration et "lift" par décile de score
    deciles = pd.qcut(proba, 10, labels=False, duplicates="drop")
    par_decile = (pd.DataFrame({"decile": deciles + 1, "proba": proba, "observe": y})
                  .groupby("decile").agg(proba_moyenne=("proba", "mean"), taux_observe=("observe", "mean"),
                                         passes=("observe", "size"))
                  .reset_index())

    rapport = {
        "cible": f"Tir de la même équipe, dans la même possession, dans les 15 secondes",
        "n_passes": int(len(passes)),
        "n_matchs": int(passes["match_id"].nunique()),
        "taux_base": round(float(y.mean()), 4),
        "modele_retenu": meilleur,
        "resultats": resultats,
        "resultats_dernier_tiers": resultats_dernier_tiers,
        "taux_base_dernier_tiers": round(float(y[dernier_tiers].mean()), 4),
        "n_dernier_tiers": int(dernier_tiers.sum()),
        "gain_contexte_360": bootstrap_gain_auc(y, proba, scores["Position seule (sans 360)"][0], groupes),
        "gain_vs_ancienne_formule": bootstrap_gain_auc(
            y, proba, scores["Ancienne formule (poids à la main)"][0], groupes),
        "deciles": par_decile.round(4).to_dict(orient="records"),
        "importance": importance_par_famille(modeles()[meilleur], X, y, groupes, n_plis),
    }
    return pd.Series(proba, index=passes.index, name="danger"), rapport


def matrice_avant(passes: pd.DataFrame) -> pd.DataFrame:
    X = passes[SITUATION_AVANT].astype(float)
    indicatrices = pd.get_dummies(passes[CATEGORIELLES_AVANT], prefix=CATEGORIELLES_AVANT, dtype=float)
    return pd.concat([X, indicatrices], axis=1)


def valeur_avant_passe(passes: pd.DataFrame, n_plis: int = 5) -> tuple[pd.Series, dict]:
    """
    Probabilité de tir dans les 15 s estimée AVANT la passe, à partir de la seule situation du passeur
    (même cible que le modèle de danger, mais sans connaître la destination de la passe).
    Validation croisée par match, comme pour le danger.
    """
    X = matrice_avant(passes)
    y = passes[CIBLE].to_numpy().astype(int)
    groupes = passes["match_id"].to_numpy()
    scores = {nom: _predictions_hors_echantillon(m, X, y, groupes, n_plis) for nom, m in modeles().items()}
    resultats = {nom: _metriques(y, s, proba=True) for nom, s in scores.items()}
    retenu = min(resultats, key=lambda n: resultats[n]["log_loss"])
    return pd.Series(scores[retenu], index=passes.index, name="valeur_avant"), {
        "modele_retenu": retenu, "resultats": resultats}


def synthese_valeur_ajoutee(passes: pd.DataFrame) -> dict:
    """Chiffres clés de la valeur ajoutée (en points de probabilité), sur le jeu courant."""
    jc = passes[passes["type_passe"] == "Jeu courant"]
    va = jc["valeur_ajoutee"] * 100
    elim = pd.cut(jc["adv_elimines"], [-1, 0, 2, 100], labels=["0", "1-2", "3+"])
    par_zone = []
    for zone, (x0, x1) in ZONES.items():
        dans_zone = (jc["x_fin"] >= x0) & (jc["x_fin"] < x1)
        for niveau in ["0", "1-2", "3+"]:
            masque = dans_zone & (elim == niveau)
            par_zone.append({"zone": zone, "elimines": niveau, "passes": int(masque.sum()),
                             "danger": round(float(jc.loc[masque, "danger"].mean() * 100), 2),
                             "valeur_ajoutee": round(float(va[masque].mean()), 2)})
    return {
        "moyenne": round(float(va.mean()), 3),
        "part_positive": round(float((va > 0).mean()), 4),
        "correlation_adv_elimines": {
            "danger": round(float(jc["danger"].corr(jc["adv_elimines"], method="spearman")), 3),
            "valeur_ajoutee": round(float(va.corr(jc["adv_elimines"], method="spearman")), 3),
        },
        "par_zone_et_elimines": par_zone,
    }
