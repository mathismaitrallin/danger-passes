"""
Application Streamlit — Dangerosité des passes, Coupe du monde 2022 (StatsBomb 360).

Les données sont préparées par construire_donnees.py (dossier data/).
Lancement local : streamlit run app_passes.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from passes.terrain import COULEURS, ECHELLE_DANGER, ajouter_passes, carte_chaleur, figure_freeze_frame, terrain

DOSSIER = Path(__file__).parent / "data"
TOP_DANGER = 0.95  # une passe est "très dangereuse" si elle est dans les 5 % les plus dangereuses du tournoi

PHASES_JEU = {
    "Regular Play": "Jeu placé", "From Counter": "Contre-attaque", "From Throw In": "Après une touche",
    "From Free Kick": "Après un coup franc", "From Corner": "Après un corner",
    "From Goal Kick": "Après un dégagement", "From Kick Off": "Après l'engagement",
    "From Keeper": "Relance du gardien", "Other": "Autre",
}
HAUTEURS = {"Ground Pass": "à ras de terre", "Low Pass": "à mi-hauteur", "High Pass": "aérienne"}

CONFIG_TERRAIN = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}
CONFIG_SIMPLE = {"displaylogo": False, "displayModeBar": False}
FOND_GRAPHIQUE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e3", family="Inter, system-ui, sans-serif"))
GRILLE = "rgba(255,255,255,0.06)"
ECHELLE_BARRES = [[0, "#2d7fb3"], [0.6, "#3dd1ff"], [1, "#6bd3a8"]]

st.set_page_config(page_title="Dangerosité des passes · Coupe du monde 2022", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}
h1 {font-weight: 800; letter-spacing: -0.5px;}
h1 span.degrade, .gros-chiffre {
  background: linear-gradient(90deg, #3dd1ff, #6bd3a8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sous-titre {color: #97a0b3; font-size: 1.02rem; max-width: 920px; margin-top: -0.4rem;}
div[data-testid="stMetric"] {
  background: #121a24; border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px; padding: 14px 18px;
}
div[data-testid="stMetricValue"] {font-weight: 700;}
.carte {
  background: #121a24; border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;
}
.carte h4 {margin: 0 0 6px 0; padding: 0; font-size: 0.95rem; color: #97a0b3; font-weight: 600;}
.gros-chiffre {font-size: 2.6rem; font-weight: 800; line-height: 1.1;}
.discret {color: #97a0b3; font-size: 0.9rem;}
.carte ul {padding-left: 1.1rem; margin: 0.4rem 0 0 0;}
.carte li {margin-bottom: 0.25rem;}
.pied {color: #6c7689; font-size: 0.85rem; margin-top: 2.5rem; border-top: 1px solid rgba(255,255,255,0.06);
       padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
@st.cache_data
def charger_donnees():
    passes = pd.read_parquet(DOSSIER / "passes.parquet")
    passes["phase_jeu_fr"] = passes["phase_jeu"].map(PHASES_JEU).fillna("Autre")
    matchs = pd.read_parquet(DOSSIER / "matchs.parquet")
    minutes = pd.read_parquet(DOSSIER / "minutes.parquet")
    rapport = json.loads((DOSSIER / "rapport_modele.json").read_text(encoding="utf-8"))
    return passes, matchs, minutes, rapport


@st.cache_resource
def charger_positions():
    return pd.read_parquet(DOSSIER / "positions_360.parquet").set_index("pid").sort_index()


passes, matchs, minutes, rapport = charger_donnees()
seuil_top = float(passes["danger"].quantile(TOP_DANGER))
taux_base = rapport["taux_base"]
# bornes des échelles de couleurs, communes à toute l'app
VMAX = {"danger": float(passes["danger"].quantile(0.995)),
        "valeur_ajoutee": float(passes["valeur_ajoutee"].quantile(0.995))}


def pct(x: float, decimales: int = 1) -> str:
    return f"{x * 100:.{decimales}f} %".replace(".", ",")


def nombre(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def dec(x: float, decimales: int = 2) -> str:
    return f"{x:.{decimales}f}".replace(".", ",")


def afficher(fig: go.Figure) -> None:
    """Affiche un graphique simple (marges automatiques pour que les libellés d'axes ne soient pas coupés)."""
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, theme=None, config=CONFIG_SIMPLE)


def carte_html(titre: str, contenu: str) -> str:
    return f"<div class='carte'><h4>{titre}</h4>{contenu}</div>"


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtres")
    options_match = {"Tous les matchs": None}
    options_match.update({f"{m.phase} · {m.libelle}": m.match_id
                          for m in matchs.sort_values("date", ascending=False).itertuples()})
    choix_match = st.selectbox("Match", list(options_match), index=1,
                               help="Par défaut : la finale Argentine – France.")
    match_id = options_match[choix_match]

    base = passes if match_id is None else passes[passes["match_id"] == match_id]
    equipe = st.selectbox("Équipe", ["Toutes les équipes"] + sorted(base["equipe"].unique()))
    if equipe != "Toutes les équipes":
        base = base[base["equipe"] == equipe]

    joueur = st.selectbox("Joueur", ["Tous les joueurs"] + sorted(base["joueur"].unique()))
    periode = st.radio("Période", ["Tout le match", "1re mi-temps", "2e mi-temps", "Prolongations"])
    cpa = st.toggle("Inclure les coups de pied arrêtés", value=False,
                    help="Corners, coups francs, touches, dégagements et engagements.")

    st.markdown("### Affichage")
    choix_mesure = st.radio(
        "Classer les passes par", ["Danger", "Valeur ajoutée"], horizontal=True,
        help="Danger : probabilité de tir dans les 15 s après la passe. Valeur ajoutée : gain de cette "
             "probabilité par rapport à la situation avant la passe (récompense aussi la progression loin du but).")
    mesure = "danger" if choix_mesure == "Danger" else "valeur_ajoutee"
    nb_affiche = st.slider("Nombre de passes sur le terrain", 10, 300, 60, step=10,
                           help="Les passes les mieux classées de la sélection sont affichées en premier.")
    if mesure == "danger":
        seuil_min = st.slider("Danger minimum (%)", 0, 60, 0, step=1,
                              help="Probabilité minimale qu'un tir suive la passe dans les 15 secondes.")
    else:
        seuil_min = st.slider("Valeur ajoutée minimum (points)", 0, 40, 0, step=1,
                              help="Gain minimal de probabilité de tir apporté par la passe.")

if not cpa:
    base = base[base["type_passe"] == "Jeu courant"]
selection = base
if joueur != "Tous les joueurs":
    selection = selection[selection["joueur"] == joueur]
selection = selection[selection["periode"].isin({"Tout le match": [1, 2, 3, 4], "1re mi-temps": [1],
                                                  "2e mi-temps": [2], "Prolongations": [3, 4]}[periode])]

affichees = (selection[selection[mesure] >= seuil_min / 100]
             .sort_values(mesure, ascending=False).head(nb_affiche))
seuil_fort = float(passes[mesure].quantile(TOP_DANGER))  # top 5 % du tournoi pour la mesure choisie


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown("<h1>Dangerosité des passes · <span class='degrade'>Coupe du monde 2022</span></h1>",
            unsafe_allow_html=True)
st.markdown(
    f"<p class='sous-titre'>Pour chacune des {nombre(rapport['n_passes'])} passes réussies du tournoi, "
    "un modèle estime la <b>probabilité qu'elle mène à un tir dans les 15 secondes</b>, à partir de "
    "l'endroit où elle arrive et de la position des joueurs au moment de la passe (données StatsBomb 360). "
    f"En moyenne, {pct(taux_base)} des passes sont suivies d'un tir. La <b>valeur ajoutée</b> compare ce "
    "danger à celui de la situation avant la passe : elle récompense aussi les passes qui font progresser "
    "l'équipe loin du but.</p>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Passes analysées", nombre(len(selection)))
c2.metric("Passes très dangereuses", nombre((selection["danger"] >= seuil_top).sum()),
          help=f"Passes dans les 5 % les plus dangereuses du tournoi (danger ≥ {pct(seuil_top, 0)}).")
c3.metric("Passes suivies d'un tir : attendu", dec(selection["danger"].sum(), 1),
          help="Somme des probabilités : nombre de passes qu'on pouvait s'attendre à voir suivies d'un tir.")
c4.metric("Passes suivies d'un tir : réel", nombre(selection["tir_15s"].sum()),
          help="Ce qui s'est vraiment passé, à comparer à la valeur attendue.")

onglet_carte, onglet_joueurs, onglet_equipes, onglet_methode = st.tabs(
    ["🗺️ Carte des passes", "👟 Joueurs", "🌍 Équipes", "🔬 Méthode & validation"])


# ---------------------------------------------------------------------------
# Onglet 1 : carte des passes
# ---------------------------------------------------------------------------
def couloir(y: pd.Series) -> pd.Series:
    """Couloir vu par l'équipe qui attaque vers la droite (y = 0 en haut = côté gauche)."""
    return pd.cut(y, [-1, 68 / 3, 2 * 68 / 3, 69], labels=["gauche", "axe", "droit"])


with onglet_carte:
    if affichees.empty:
        st.info("Aucune passe ne correspond à ces filtres. Essayez de baisser le seuil minimum.")
    else:
        carte = ajouter_passes(terrain(hauteur=640, barre_couleurs=True), affichees, vmax=VMAX[mesure],
                               mesure=mesure)
        st.plotly_chart(carte, theme=None, config=CONFIG_TERRAIN)
        quoi = "les plus dangereuses" if mesure == "danger" else "qui ajoutent le plus de valeur"
        st.caption(f"Les {len(affichees)} passes {quoi} de la sélection. Toutes les passes sont "
                   "orientées vers la droite (sens d'attaque de l'équipe qui passe). Survolez une flèche pour "
                   "le détail.")

        # --- Lecture rapide ---
        fortes = selection[selection[mesure] >= seuil_fort]
        nom_fortes = "passes très dangereuses" if mesure == "danger" else "passes à forte valeur ajoutée"
        st.markdown("#### Lecture rapide")
        l1, l2, l3 = st.columns(3)
        if len(fortes) >= 5:
            part = couloir(fortes["y_debut"]).value_counts(normalize=True)
            l1.markdown(carte_html(
                f"D'où partent les {nom_fortes}",
                f"<span class='gros-chiffre'>{pct(part.max(), 0)}</span><br><span class='discret'>depuis le "
                f"couloir {part.idxmax()} (gauche {pct(part.get('gauche', 0), 0)} · axe "
                f"{pct(part.get('axe', 0), 0)} · droit {pct(part.get('droit', 0), 0)})</span>"),
                unsafe_allow_html=True)
            if mesure == "danger":
                l2.markdown(carte_html(
                    "Où arrivent-elles",
                    f"<span class='gros-chiffre'>{pct(fortes['dans_surface_fin'].mean(), 0)}</span><br>"
                    f"<span class='discret'>dans la surface · {pct((fortes['derriere_defense'] > 0).mean(), 0)} "
                    "dans le dos du dernier défenseur visible</span>"),
                    unsafe_allow_html=True)
            else:
                l2.markdown(carte_html(
                    "Comment elles font progresser",
                    f"<span class='gros-chiffre'>{pct((fortes['adv_elimines'] >= 3).mean(), 0)}</span><br>"
                    "<span class='discret'>éliminent au moins 3 adversaires · progression moyenne de "
                    f"{fortes['progression'].mean():.0f} m vers le but</span>"),
                    unsafe_allow_html=True)
        else:
            texte = f"<span class='discret'>Pas assez de {nom_fortes} dans la sélection.</span>"
            l1.markdown(carte_html(f"D'où partent les {nom_fortes}", texte), unsafe_allow_html=True)
            l2.markdown(carte_html("Où arrivent-elles", texte), unsafe_allow_html=True)
        meilleurs = selection.groupby(["joueur", "equipe"])[mesure].sum().sort_values(ascending=False).head(3)
        lignes = "".join(f"<li><b>{j}</b> ({e}) · {dec(v)}</li>" for (j, e), v in meilleurs.items())
        titre_l3 = "Passeurs qui créent le plus de danger" if mesure == "danger" else \
            "Passeurs qui ajoutent le plus de valeur"
        detail_l3 = "somme des probabilités de tir de leurs passes" if mesure == "danger" else \
            "somme des gains de probabilité de leurs passes"
        l3.markdown(carte_html(titre_l3, f"<ul>{lignes}</ul><span class='discret'>{detail_l3}</span>"),
                    unsafe_allow_html=True)

        # --- Revoir une passe ---
        st.markdown("#### Revoir une passe")
        st.caption("Cliquez sur une ligne pour voir la position des joueurs au moment de la passe.")
        tableau = affichees.assign(
            Minute=affichees["minute"].astype(str) + "'",
            Suite=np.where(affichees["but_15s"], "⚽ But", np.where(affichees["tir_15s"], "Tir", "—")),
            Danger=affichees["danger"] * 100,
            Valeur=affichees["valeur_ajoutee"] * 100,
        ).rename(columns={"joueur": "Passeur", "receveur": "Receveur", "equipe": "Équipe",
                          "adversaire": "Adversaire", "phase_jeu_fr": "Phase de jeu"})
        evenement = st.dataframe(
            tableau[["Minute", "Passeur", "Receveur", "Équipe", "Adversaire", "Phase de jeu", "Danger", "Valeur",
                     "Suite"]],
            hide_index=True, height=290, on_select="rerun", selection_mode="single-row", key="table_passes",
            column_config={
                "Danger": st.column_config.ProgressColumn(
                    "Danger", format="%.1f %%", min_value=0, max_value=VMAX["danger"] * 100,
                    help="Probabilité qu'un tir suive dans les 15 secondes"),
                "Valeur": st.column_config.NumberColumn(
                    "Valeur ajoutée", format="%+.1f pts",
                    help="Gain de probabilité de tir par rapport à la situation avant la passe"),
            },
        )
        lignes_choisies = evenement.selection.rows
        passe = affichees.iloc[lignes_choisies[0] if lignes_choisies else 0]

        positions = charger_positions()
        joueurs_ff = positions.loc[[passe["pid"]]] if passe["pid"] in positions.index else positions.iloc[0:0]

        g, d = st.columns([2.3, 1])
        with g:
            st.plotly_chart(figure_freeze_frame(passe, joueurs_ff, VMAX["danger"]), theme=None, config=CONFIG_TERRAIN,
                            key="freeze_frame")
            st.caption("Zone en pointillés : partie du terrain visible à la caméra. Les joueurs hors champ ne "
                       "sont pas connus (limite des données 360, issues des images TV).")
        with d:
            if passe["but_15s"]:
                suite = f"⚽ <b>But</b> dans les 15 secondes (xG du tir : {dec(passe['xg_15s'])})"
            elif passe["tir_15s"]:
                suite = f"Un <b>tir</b> a suivi (xG : {dec(passe['xg_15s'])})"
            else:
                suite = "Pas de tir dans les 15 secondes"
            technique = ", ".join(t for t, ok in [("en profondeur", passe["en_profondeur"]),
                                                  ("centre", passe["centre"]),
                                                  ("passe en retrait", passe["passe_en_retrait"]),
                                                  ("renversement", passe["renversement"])] if ok)
            dos = (f"oui, {passe['derriere_defense']:.0f} m derrière le dernier défenseur visible"
                   if passe["derriere_defense"] > 0 else "non")
            st.markdown(
                carte_html(f"{passe['joueur']} → {passe['receveur']}",
                           f"<span class='discret'>{passe['equipe']} vs {passe['adversaire']} · "
                           f"{passe['minute']}'</span><br><br><span class='gros-chiffre'>{pct(passe['danger'])}"
                           f"</span><br><span class='discret'>de chances de tir dans les 15 s, soit "
                           f"<b>{dec(passe['danger'] / taux_base, 1)}×</b> la moyenne d'une passe</span>")
                + carte_html("Valeur ajoutée",
                             f"<span class='gros-chiffre'>{passe['valeur_ajoutee'] * 100:+.1f} pts".replace(".", ",")
                             + f"</span><br><span class='discret'>la probabilité de tir passe de "
                             f"<b>{pct(passe['valeur_avant'])}</b> avant la passe à <b>{pct(passe['danger'])}</b> "
                             "après</span>")
                + carte_html("Le contexte de la passe", f"""<ul>
<li>Arrivée à <b>{passe['dist_but_fin']:.0f} m</b> du but, angle de tir de {passe['angle_but_fin']:.0f}°</li>
<li><b>{passe['adv_elimines']}</b> adversaire(s) éliminé(s)</li>
<li><b>{passe['defenseurs_restants']}</b> défenseur(s) encore entre le ballon et le but</li>
<li>Dans le dos de la défense : {dos}</li>
<li>Adversaire le plus proche du receveur : {dec(passe['dist_adv_fin'], 1)} m</li>
<li>Passe {HAUTEURS.get(passe['hauteur'], '')}{', ' + technique if technique else ''} ·
{passe['phase_jeu_fr'].lower()}</li></ul>""")
                + carte_html("Suite de l'action", suite),
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Onglet 2 : joueurs
# ---------------------------------------------------------------------------
with onglet_joueurs:
    st.markdown("#### Qui crée le plus de danger par la passe ?")
    st.caption("Calculé sur la sélection de match, d'équipe et de type de passe (toutes périodes). "
               "Le « danger créé » additionne les probabilités de tir de toutes les passes d'un joueur : "
               "c'est le nombre de ses passes qu'on pouvait s'attendre à voir suivies d'un tir. La « valeur "
               "ajoutée » additionne ce que chaque passe a fait gagner (ou perdre) par rapport à la situation "
               "de départ : elle met en avant les joueurs qui font progresser l'équipe, même loin du but. Une valeur "
               "négative n'est pas un défaut : faire tourner le ballon en sécurité (Rodri, Gavi) dégrade "
               "légèrement la situation immédiate mais peut servir le jeu de possession. "
               "Les joueurs des équipes qui ont beaucoup le ballon (Espagne, Allemagne) sont avantagés par le volume.")

    mins = minutes if match_id is None else minutes[minutes["match_id"] == match_id]
    if equipe != "Toutes les équipes":
        mins = mins[mins["equipe"] == equipe]
    mins = mins.groupby("joueur_id", as_index=False)["minutes"].sum()

    agr = (base.groupby(["joueur_id", "joueur", "equipe"], as_index=False)
           .agg(passes=("danger", "size"), danger=("danger", "sum"), valeur=("valeur_ajoutee", "sum"),
                tres_dangereuses=("danger", lambda s: int((s >= seuil_top).sum())),
                meilleure=("danger", "max"), tirs=("tir_15s", "sum"))
           .merge(mins, on="joueur_id", how="left"))
    agr["minutes"] = agr["minutes"].fillna(0)
    agr["danger_90"] = agr["danger"] / agr["minutes"].clip(lower=1) * 90
    agr["valeur_90"] = agr["valeur"] / agr["minutes"].clip(lower=1) * 90

    m1, m2 = st.columns(2)
    min_minutes = m1.slider("Minutes jouées minimum", 0, 600 if match_id is None else 120,
                            270 if match_id is None else 30, step=10)
    criteres = {"Danger créé / 90 min": "danger_90", "Danger créé (total)": "danger",
                "Valeur ajoutée / 90 min": "valeur_90", "Valeur ajoutée (total)": "valeur",
                "Passes très dangereuses": "tres_dangereuses"}
    critere = m2.radio("Classer par", list(criteres), horizontal=True, index=0 if mesure == "danger" else 2)
    cle = criteres[critere]
    classement = agr[agr["minutes"] >= min_minutes].sort_values(cle, ascending=False)

    if classement.empty:
        st.info("Aucun joueur ne correspond. Baissez le nombre de minutes minimum.")
    else:
        top = classement.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top[cle], y=top["joueur"] + "  ·  " + top["equipe"], orientation="h",
            marker=dict(color=top[cle], colorscale=ECHELLE_BARRES, line=dict(width=0)),
            text=[dec(v) if cle != "tres_dangereuses" else f"{v:.0f}" for v in top[cle]],
            textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>%{x:.2f}<extra></extra>",
        ))
        fig.update_layout(height=540, margin=dict(l=8, r=40, t=10, b=40), **FOND_GRAPHIQUE,
                          xaxis=dict(gridcolor=GRILLE, zeroline=False, title=critere),
                          yaxis=dict(showgrid=False, automargin=True))
        afficher(fig)

        st.dataframe(
            classement.rename(columns={"joueur": "Joueur", "equipe": "Équipe", "minutes": "Minutes",
                                       "passes": "Passes", "danger": "Danger créé",
                                       "danger_90": "Danger / 90", "valeur": "Valeur ajoutée",
                                       "valeur_90": "Valeur / 90", "tres_dangereuses": "Très dangereuses",
                                       "meilleure": "Meilleure passe", "tirs": "Suivies d'un tir"}),
            hide_index=True, height=380,
            column_order=["Joueur", "Équipe", "Minutes", "Passes", "Danger créé", "Danger / 90",
                          "Valeur ajoutée", "Valeur / 90", "Très dangereuses", "Suivies d'un tir",
                          "Meilleure passe"],
            column_config={
                "Minutes": st.column_config.NumberColumn(format="%d"),
                "Danger créé": st.column_config.NumberColumn(format="%.2f",
                                                             help="Somme des probabilités de tir de ses passes"),
                "Danger / 90": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0, max_value=float(max(classement["danger_90"].max(), 0.01))),
                "Valeur ajoutée": st.column_config.NumberColumn(
                    format="%+.2f", help="Somme des gains de probabilité de tir apportés par ses passes"),
                "Valeur / 90": st.column_config.NumberColumn(format="%+.2f"),
                "Meilleure passe": st.column_config.NumberColumn(
                    format="%.2f", help="Probabilité de tir après sa passe la plus dangereuse"),
            },
        )


# ---------------------------------------------------------------------------
# Onglet 3 : équipes
# ---------------------------------------------------------------------------
with onglet_equipes:
    st.markdown("#### Danger créé et concédé par match")
    st.caption("Sur tout le tournoi, " + ("tous types de passes." if cpa else "jeu courant uniquement."))
    pe = passes if cpa else passes[passes["type_passe"] == "Jeu courant"]
    nb_matchs = pd.concat([matchs["domicile"], matchs["exterieur"]]).value_counts()
    eq = pd.DataFrame({"cree": pe.groupby("equipe")["danger"].sum() / nb_matchs,
                       "concede": pe.groupby("adversaire")["danger"].sum() / nb_matchs,
                       "matchs": nb_matchs}).dropna()

    fig = go.Figure(go.Scatter(
        x=eq["cree"], y=eq["concede"], mode="markers+text", text=eq.index, textposition="top center",
        textfont=dict(size=11, color="#c9d4e3"),
        marker=dict(size=8 + eq["matchs"] * 1.6, color=eq["cree"] - eq["concede"],
                    colorscale=[[0, COULEURS["defense"]], [0.5, "#97a0b3"], [1, COULEURS["attaque"]]],
                    line=dict(color="#0b0f13", width=1)),
        customdata=eq[["matchs"]],
        hovertemplate="<b>%{text}</b><br>Créé : %{x:.2f} / match<br>Concédé : %{y:.2f} / match"
                      "<br>%{customdata[0]} matchs<extra></extra>",
    ))
    fig.add_vline(x=eq["cree"].median(), line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
    fig.add_hline(y=eq["concede"].median(), line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
    fig.update_layout(height=580, margin=dict(l=10, r=10, t=20, b=10), **FOND_GRAPHIQUE,
                      xaxis=dict(title="Danger créé par match (somme des probabilités)", gridcolor=GRILLE,
                                 zeroline=False, automargin=True),
                      yaxis=dict(title="Danger concédé par match", gridcolor=GRILLE, zeroline=False,
                                 autorange="reversed", automargin=True))
    afficher(fig)
    st.caption("Axe vertical inversé : en haut à droite, les équipes dangereuses et solides ; en bas à gauche, "
               "celles qui créent peu et concèdent beaucoup. Pointillés : médianes. "
               "Taille des points = nombre de matchs joués.")

    st.markdown("#### Profil d'une équipe")
    equipes_triees = sorted(eq.index)
    defaut = equipe if equipe in equipes_triees else "Argentine"
    choix = st.selectbox("Équipe à analyser", equipes_triees, index=equipes_triees.index(defaut))
    attaque = pe[(pe["equipe"] == choix) & (pe["danger"] >= seuil_top)]
    defense = pe[(pe["adversaire"] == choix) & (pe["danger"] >= seuil_top)]
    a, b = st.columns(2)
    with a:
        st.plotly_chart(carte_chaleur(attaque["x_debut"], attaque["y_debut"], poids=attaque["danger"],
                                      titre="D'où partent ses passes très dangereuses", hauteur=430),
                        theme=None, config=CONFIG_SIMPLE)
        st.caption(f"{len(attaque)} passes très dangereuses de : {choix} · attaque vers la droite.")
    with b:
        st.plotly_chart(carte_chaleur(defense["x_fin"], defense["y_fin"], poids=defense["danger"],
                                      titre="Où ses adversaires font arriver leurs passes très dangereuses",
                                      couleur="255,107,107", hauteur=430),
                        theme=None, config=CONFIG_SIMPLE)
        st.caption(f"{len(defense)} passes très dangereuses concédées · le but de : {choix} est à droite.")


# ---------------------------------------------------------------------------
# Onglet 4 : méthode et validation
# ---------------------------------------------------------------------------
with onglet_methode:
    res, res3 = rapport["resultats"], rapport["resultats_dernier_tiers"]
    retenu = rapport["modele_retenu"]
    gain360, gain_ancien = rapport["gain_contexte_360"], rapport["gain_vs_ancienne_formule"]
    ancienne = "Ancienne formule (poids à la main)"

    st.markdown("#### La question")
    st.markdown(
        "Toutes les passes ne se valent pas : une passe latérale entre défenseurs ne crée rien, une passe qui "
        "élimine trois adversaires et arrive seule face au but change le match. L'objectif est de **mesurer "
        "ce danger de façon objective**, puis de vérifier que la mesure tient la route.")

    st.markdown("#### Les données")
    st.markdown(
        f"- **StatsBomb Open Data**, Coupe du monde 2022 : {rapport['n_matchs']} matchs, chaque action annotée.\n"
        "- **StatsBomb 360** : pour chaque action, la position de tous les joueurs visibles à l'image. "
        "C'est ce qui permet de savoir combien d'adversaires une passe élimine ou quel espace a le receveur.\n"
        f"- {nombre(rapport['n_passes'])} passes réussies retenues, soit {pct(rapport['couverture_360'], 0)} "
        "des passes réussies (les autres n'ont pas d'image 360, par exemple pendant un ralenti).")

    st.markdown("#### Ce que le modèle prédit")
    st.markdown(
        f"La cible est simple et vérifiable : **la même équipe tire-t-elle dans les 15 secondes, sans perdre "
        f"le ballon ?** C'est le cas pour {pct(taux_base)} des passes. Le score de danger d'une passe est la "
        "probabilité estimée de cet événement. Les variables décrivent uniquement ce qu'on sait au moment de "
        "la passe, jamais la suite de l'action.")

    st.markdown("#### Validation")
    st.markdown(
        "Le modèle est évalué en **validation croisée par match** : il est entraîné sur 4/5 des matchs et "
        "noté sur les matchs restants, cinq fois de suite. Chaque score affiché dans l'application vient donc "
        "d'un modèle qui n'a jamais vu le match en question. L'**AUC** mesure la capacité à bien classer les "
        "passes (0,5 = hasard, 1 = parfait).")

    ordre = [ancienne, "Distance au but seule", "Position seule (sans 360)", "Gradient boosting",
             "Régression logistique"]
    noms = [n + " ✓" if n == retenu else n for n in ordre]
    fig = go.Figure()
    for jeu, nom_jeu, couleur in [(res, "Toutes les passes", "#3dd1ff"),
                                  (res3, "Passes vers le dernier tiers", "#6bd3a8")]:
        fig.add_trace(go.Bar(y=noms, x=[jeu[n]["auc"] for n in ordre], orientation="h", name=nom_jeu,
                             marker_color=couleur, text=[dec(jeu[n]["auc"], 3) for n in ordre],
                             textposition="outside", cliponaxis=False))
    fig.update_layout(barmode="group", height=430, margin=dict(l=8, r=50, t=10, b=10), **FOND_GRAPHIQUE,
                      xaxis=dict(range=[0.5, 0.9], title="AUC (validation par match)", gridcolor=GRILLE),
                      yaxis=dict(autorange="reversed", automargin=True), legend=dict(orientation="h", y=1.1, x=0))
    afficher(fig)

    st.markdown(carte_html("Ce qu'il faut retenir", f"""<ul>
<li><b>La première version du projet</b> (poids choisis à la main) ne distinguait presque pas les passes
dangereuses dans le dernier tiers : AUC de {dec(res3[ancienne]['auc'])}, contre {dec(res3[retenu]['auc'])} pour
le modèle. Sur toutes les passes, le gain est de +{dec(gain_ancien['gain'], 3)} (intervalle de confiance à 95 % :
{dec(gain_ancien['ic95'][0], 3)} à {dec(gain_ancien['ic95'][1], 3)}).</li>
<li><b>L'endroit où arrive la passe explique l'essentiel du danger</b> : la distance au but seule atteint déjà
{dec(res['Distance au but seule']['auc'], 3)}.</li>
<li><b>Le contexte 360 apporte un gain réel mais modeste</b> : +{dec(gain360['gain'], 3)} d'AUC par rapport à
un modèle qui ne connaît que la position (IC 95 % : {dec(gain360['ic95'][0], 3)} à {dec(gain360['ic95'][1], 3)}).
C'est attendu : les images TV ne montrent pas tous les joueurs, et la suite d'une action dépend de bien d'autres
choses que la passe.</li>
<li>La <b>{retenu.lower()}</b> est retenue : aussi précise que le gradient boosting, mieux calibrée
(log loss {dec(res[retenu]['log_loss'], 4)} contre {dec(res['Gradient boosting']['log_loss'], 4)}) et plus
facile à expliquer.</li></ul>"""), unsafe_allow_html=True)

    g, d = st.columns(2)
    with g:
        st.markdown("##### Les probabilités sont-elles fiables ?")
        deciles = pd.DataFrame(rapport["deciles"])
        lim = float(max(deciles["proba_moyenne"].max(), deciles["taux_observe"].max())) * 110
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", hoverinfo="skip",
                                 line=dict(color="rgba(255,255,255,0.25)", dash="dot")))
        fig.add_trace(go.Scatter(x=deciles["proba_moyenne"] * 100, y=deciles["taux_observe"] * 100,
                                 mode="lines+markers", line=dict(color="#3dd1ff", width=2.5),
                                 marker=dict(size=9, color="#3dd1ff"), customdata=deciles[["decile", "passes"]],
                                 hovertemplate="Tranche %{customdata[0]} (%{customdata[1]} passes)<br>"
                                               "Prédit : %{x:.1f} %<br>Observé : %{y:.1f} %<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=8, r=8, t=10, b=10), **FOND_GRAPHIQUE, showlegend=False,
                          xaxis=dict(title="Probabilité prédite (%)", gridcolor=GRILLE),
                          yaxis=dict(title="Fréquence réelle de tir (%)", gridcolor=GRILLE))
        afficher(fig)
        dernier = deciles.iloc[-1]
        st.caption(f"Passes regroupées en 10 tranches de danger. Les points suivent la diagonale : pour les "
                   f"10 % de passes les plus dangereuses, le modèle annonce {dernier['proba_moyenne'] * 100:.0f} % "
                   f"en moyenne, et un tir suit réellement dans {dernier['taux_observe'] * 100:.0f} % des cas.")
    with d:
        st.markdown("##### Quelles informations comptent le plus ?")
        imp = pd.Series(rapport["importance"]).clip(lower=0).sort_values()
        fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h",
                               marker=dict(color=imp.values, colorscale=ECHELLE_BARRES),
                               text=[dec(v, 2) for v in imp.values * 100], textposition="outside",
                               cliponaxis=False,
                               hovertemplate="%{y}<br>Perte d'AUC : %{x:.2f} point(s)<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=8, r=45, t=10, b=10), **FOND_GRAPHIQUE,
                          xaxis=dict(title="Perte d'AUC si on brouille l'information (points)",
                                     gridcolor=GRILLE), yaxis=dict(automargin=True))
        afficher(fig)
        st.caption("Importance par permutation : on mélange une famille de variables et "
                   "on mesure combien le modèle se dégrade.")

    st.markdown("##### Où une passe devient-elle dangereuse ?")
    jeu_courant = passes[passes["type_passe"] == "Jeu courant"]
    st.plotly_chart(carte_chaleur(jeu_courant["x_fin"], jeu_courant["y_fin"], moyenne=True,
                                  valeurs=jeu_courant["danger"], nx=21, ny=14, hauteur=520),
                    theme=None, config=CONFIG_SIMPLE)
    st.caption("Danger moyen selon la zone d'arrivée de la passe (jeu courant, zones d'au moins 30 passes).")

    st.markdown("#### La valeur ajoutée : récompenser aussi la progression")
    avant, va = rapport["valeur_avant"], rapport["valeur_ajoutee"]
    st.markdown(
        "Le danger mesure la **proximité d'un tir** : une passe qui casse une ligne au milieu de terrain reste "
        "loin du but, donc peu dangereuse, même si elle fait beaucoup progresser l'équipe. Pour la récompenser, "
        "un second modèle estime la probabilité de tir **avant la passe**, à partir de la seule situation du "
        "passeur (position du ballon, pression, défenseurs entre lui et le but, phase de jeu), sans connaître la "
        "destination. La **valeur ajoutée** est la différence :\n\n"
        "> valeur ajoutée = danger après la passe − probabilité de tir avant la passe\n\n"
        f"Elle s'exprime en points de probabilité et vaut zéro en moyenne ({dec(va['moyenne'], 2)} pt) : une "
        "passe positive améliore la situation, une passe négative (une passe en retrait, par exemple) la dégrade. "
        f"Le modèle « avant la passe » ({avant['modele_retenu'].lower()}) atteint une AUC de "
        f"{dec(avant['resultats'][avant['modele_retenu']]['auc'], 3)} en validation par match.")

    par_zone = pd.DataFrame(va["par_zone_et_elimines"])
    fig = go.Figure()
    for niveau, couleur in [("0", "#2d7fb3"), ("1-2", "#3dd1ff"), ("3+", "#6bd3a8")]:
        sous = par_zone[par_zone["elimines"] == niveau]
        fig.add_trace(go.Bar(x=sous["zone"], y=sous["valeur_ajoutee"], name=f"{niveau} adversaire(s) éliminé(s)",
                             marker_color=couleur, customdata=sous[["passes", "danger"]],
                             text=[f"{v:+.1f}".replace(".", ",") for v in sous["valeur_ajoutee"]],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>Valeur ajoutée moyenne : %{y:+.2f} pts<br>"
                                           "Danger moyen : %{customdata[1]:.1f} %<br>%{customdata[0]} passes"
                                           "<extra></extra>"))
    fig.update_layout(barmode="group", height=420, margin=dict(l=8, r=8, t=10, b=10), **FOND_GRAPHIQUE,
                      xaxis=dict(title="Zone d'arrivée de la passe"),
                      yaxis=dict(title="Valeur ajoutée moyenne (points)", gridcolor=GRILLE, zeroline=True,
                                 zerolinecolor="rgba(255,255,255,0.3)"),
                      legend=dict(orientation="h", y=1.1, x=0))
    afficher(fig)
    correlation = va["correlation_adv_elimines"]
    st.caption("Valeur ajoutée moyenne selon la zone d'arrivée et le nombre d'adversaires éliminés (jeu courant). "
               f"Corrélation (Spearman) avec le nombre d'adversaires éliminés : {dec(correlation['danger'], 2)} "
               f"pour le danger, {dec(correlation['valeur_ajoutee'], 2)} pour la valeur ajoutée. Limite : dans sa "
               "propre moitié, même une passe qui élimine 3 adversaires ajoute peu, car la probabilité de tir y "
               "reste très faible avant comme après la passe.")

    st.markdown("#### Limites")
    st.markdown(
        "- **Joueurs hors champ** : les données 360 viennent des images TV. Un défenseur hors cadre n'est pas "
        "compté, d'où la variable « adversaires visibles » qui sert de contrôle.\n"
        "- **Une photo, pas un film** : les positions sont celles du moment de la passe, pas de la réception.\n"
        "- **Un seul tournoi** : 64 matchs. Le modèle reste à tester sur d'autres compétitions (Euro 2024, "
        "Ligue 1) avant d'en tirer des conclusions générales.\n"
        "- **Seules les passes réussies** sont notées : le risque d'une passe tentée n'est pas mesuré.")

st.markdown(
    "<div class='pied'>Projet réalisé par <b>Mathis Maitrallin</b>, Data Scientist. "
    "Données : <a href='https://github.com/statsbomb/open-data' target='_blank'>StatsBomb Open Data</a> "
    "(Coupe du monde 2022, 360). Python · pandas · scikit-learn · Plotly · Streamlit.</div>",
    unsafe_allow_html=True)
