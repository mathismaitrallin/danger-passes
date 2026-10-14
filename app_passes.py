"""
Application Streamlit — Dangerosité des passes, Coupe du monde 2022 (StatsBomb 360).

Les données sont préparées par construire_donnees.py (dossier data/).
Lancement local : streamlit run app_passes.py
Version anglaise : ajouter ?lang=en à l'adresse (ou boutons FR | EN en haut de la barre latérale).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from passes.langue import (EQUIPES_EN, HAUTEURS, MODELES_EN, PHASES_JEU, PHASES_TOURNOI_EN, VARIABLES_EN, ZONES_EN,
                           dec, langue, libelle, nombre, pct, points, selecteur_langue, tr)
from passes.terrain import COULEURS, ECHELLE_DANGER, ajouter_passes, carte_chaleur, figure_freeze_frame, terrain

DOSSIER = Path(__file__).parent / "data"
TOP_DANGER = 0.95  # une passe est "très dangereuse" si elle est dans les 5 % les plus dangereuses du tournoi
LANGUE = langue()

CONFIG_TERRAIN = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}
CONFIG_SIMPLE = {"displaylogo": False, "displayModeBar": False}
FOND_GRAPHIQUE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d4e3", family="Inter, system-ui, sans-serif"))
GRILLE = "rgba(255,255,255,0.06)"
ECHELLE_BARRES = [[0, "#2d7fb3"], [0.6, "#3dd1ff"], [1, "#6bd3a8"]]

st.set_page_config(page_title=tr("Dangerosité des passes · Coupe du monde 2022", "Pass danger · 2022 World Cup"),
                   page_icon="⚽", layout="wide")

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
def charger_donnees(lang: str):
    """Données de l'application ; en anglais, les noms d'équipes et de phases sont traduits pour l'affichage."""
    passes = pd.read_parquet(DOSSIER / "passes.parquet")
    passes["phase_jeu_lib"] = passes["phase_jeu"].map(PHASES_JEU[lang]).fillna(PHASES_JEU[lang]["Other"])
    matchs = pd.read_parquet(DOSSIER / "matchs.parquet")
    minutes = pd.read_parquet(DOSSIER / "minutes.parquet")
    rapport = json.loads((DOSSIER / "rapport_modele.json").read_text(encoding="utf-8"))
    if lang == "en":
        for col in ("equipe", "adversaire"):
            passes[col] = passes[col].replace(EQUIPES_EN)
        minutes["equipe"] = minutes["equipe"].replace(EQUIPES_EN)
        matchs["domicile"] = matchs["domicile"].replace(EQUIPES_EN)
        matchs["exterieur"] = matchs["exterieur"].replace(EQUIPES_EN)
        matchs["phase"] = matchs["phase"].replace(PHASES_TOURNOI_EN)
        matchs["libelle"] = matchs["domicile"] + " " + matchs["score"] + " " + matchs["exterieur"]
    return passes, matchs, minutes, rapport


@st.cache_resource
def charger_positions():
    return pd.read_parquet(DOSSIER / "positions_360.parquet").set_index("pid").sort_index()


passes, matchs, minutes, rapport = charger_donnees(LANGUE)
seuil_top = float(passes["danger"].quantile(TOP_DANGER))
taux_base = rapport["taux_base"]
# bornes des échelles de couleurs, communes à toute l'app
VMAX = {"danger": float(passes["danger"].quantile(0.995)),
        "valeur_ajoutee": float(passes["valeur_ajoutee"].quantile(0.995))}
FORMAT_PCT = tr("%.1f %%", "%.1f%%")   # format des colonnes en pourcentage des tableaux


def afficher(fig: go.Figure) -> None:
    """Affiche un graphique simple (marges automatiques pour que les libellés d'axes ne soient pas coupés)."""
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, theme=None, config=CONFIG_SIMPLE)


def carte_html(titre: str, contenu: str) -> str:
    return f"<div class='carte'><h4>{titre}</h4>{contenu}</div>"


def modele(nom: str) -> str:
    return libelle(nom, MODELES_EN)


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
TOUTES_EQUIPES, TOUS_JOUEURS = tr("Toutes les équipes", "All teams"), tr("Tous les joueurs", "All players")
PERIODES = {"tout": [1, 2, 3, 4], "mt1": [1], "mt2": [2], "prolongations": [3, 4]}
NOMS_PERIODES = {"tout": tr("Tout le match", "Full match"), "mt1": tr("1re mi-temps", "1st half"),
                 "mt2": tr("2e mi-temps", "2nd half"), "prolongations": tr("Prolongations", "Extra time")}
NOMS_MESURES = {"danger": "Danger", "valeur_ajoutee": tr("Valeur ajoutée", "Value added")}

with st.sidebar:
    selecteur_langue()
    st.markdown(tr("### Filtres", "### Filters"))
    options_match = {tr("Tous les matchs", "All matches"): None}
    options_match.update({f"{m.phase} · {m.libelle}": m.match_id
                          for m in matchs.sort_values("date", ascending=False).itertuples()})
    choix_match = st.selectbox("Match", list(options_match), index=1,
                               help=tr("Par défaut : la finale Argentine – France.",
                                       "Default: the Argentina – France final."))
    match_id = options_match[choix_match]

    base = passes if match_id is None else passes[passes["match_id"] == match_id]
    equipe = st.selectbox(tr("Équipe", "Team"), [TOUTES_EQUIPES] + sorted(base["equipe"].unique()))
    if equipe != TOUTES_EQUIPES:
        base = base[base["equipe"] == equipe]

    joueur = st.selectbox(tr("Joueur", "Player"), [TOUS_JOUEURS] + sorted(base["joueur"].unique()))
    periode = st.radio(tr("Période", "Period"), list(PERIODES), format_func=NOMS_PERIODES.get)
    cpa = st.toggle(tr("Inclure les coups de pied arrêtés", "Include set pieces"), value=False,
                    help=tr("Corners, coups francs, touches, dégagements et engagements.",
                            "Corners, free kicks, throw-ins, goal kicks and kick-offs."))

    st.markdown(tr("### Affichage", "### Display"))
    mesure = st.radio(
        tr("Classer les passes par", "Rank passes by"), list(NOMS_MESURES), format_func=NOMS_MESURES.get,
        horizontal=True,
        help=tr("Danger : probabilité de tir dans les 15 s après la passe. Valeur ajoutée : gain de cette "
                "probabilité par rapport à la situation avant la passe (récompense aussi la progression loin du but).",
                "Danger: probability of a shot within 15 s of the pass. Value added: the gain in that probability "
                "compared with the situation before the pass (also rewards progress far from goal)."))
    nb_affiche = st.slider(tr("Nombre de passes sur le terrain", "Number of passes on the pitch"), 10, 300, 60,
                           step=10, help=tr("Les passes les mieux classées de la sélection sont affichées en premier.",
                                            "The highest-ranked passes in the selection are shown first."))
    if mesure == "danger":
        seuil_min = st.slider(tr("Danger minimum (%)", "Minimum danger (%)"), 0, 60, 0, step=1,
                              help=tr("Probabilité minimale qu'un tir suive la passe dans les 15 secondes.",
                                      "Minimum probability that a shot follows the pass within 15 seconds."))
    else:
        seuil_min = st.slider(tr("Valeur ajoutée minimum (points)", "Minimum value added (points)"), 0, 40, 0,
                              step=1, help=tr("Gain minimal de probabilité de tir apporté par la passe.",
                                              "Minimum gain in shot probability brought by the pass."))

if not cpa:
    base = base[base["type_passe"] == "Jeu courant"]
selection = base
if joueur != TOUS_JOUEURS:
    selection = selection[selection["joueur"] == joueur]
selection = selection[selection["periode"].isin(PERIODES[periode])]

affichees = (selection[selection[mesure] >= seuil_min / 100]
             .sort_values(mesure, ascending=False).head(nb_affiche))
seuil_fort = float(passes[mesure].quantile(TOP_DANGER))  # top 5 % du tournoi pour la mesure choisie


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown(tr("<h1>Dangerosité des passes · <span class='degrade'>Coupe du monde 2022</span></h1>",
               "<h1>Pass danger · <span class='degrade'>2022 World Cup</span></h1>"),
            unsafe_allow_html=True)
st.markdown(tr(
    f"<p class='sous-titre'>Pour chacune des {nombre(rapport['n_passes'])} passes réussies du tournoi, "
    "un modèle estime la <b>probabilité qu'elle mène à un tir dans les 15 secondes</b>, à partir de "
    "l'endroit où elle arrive et de la position des joueurs au moment de la passe (données StatsBomb 360). "
    f"En moyenne, {pct(taux_base)} des passes sont suivies d'un tir. La <b>valeur ajoutée</b> compare ce "
    "danger à celui de la situation avant la passe : elle récompense aussi les passes qui font progresser "
    "l'équipe loin du but.</p>",
    f"<p class='sous-titre'>For each of the {nombre(rapport['n_passes'])} completed passes of the tournament, "
    "a model estimates the <b>probability that it leads to a shot within 15 seconds</b>, based on "
    "where the pass ends and on the players' positions at the moment of the pass (StatsBomb 360 data). "
    f"On average, {pct(taux_base)} of passes are followed by a shot. <b>Value added</b> compares this "
    "danger with that of the situation before the pass: it also rewards passes that move the team forward "
    "far from goal.</p>"),
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(tr("Passes analysées", "Passes analysed"), nombre(len(selection)))
c2.metric(tr("Passes très dangereuses", "Very dangerous passes"), nombre((selection["danger"] >= seuil_top).sum()),
          help=tr(f"Passes dans les 5 % les plus dangereuses du tournoi (danger ≥ {pct(seuil_top, 0)}).",
                  f"Passes in the tournament's 5% most dangerous (danger ≥ {pct(seuil_top, 0)})."))
c3.metric(tr("Passes suivies d'un tir : attendu", "Passes followed by a shot: expected"),
          dec(selection["danger"].sum(), 1),
          help=tr("Somme des probabilités : nombre de passes qu'on pouvait s'attendre à voir suivies d'un tir.",
                  "Sum of probabilities: the number of passes that could be expected to be followed by a shot."))
c4.metric(tr("Passes suivies d'un tir : réel", "Passes followed by a shot: actual"),
          nombre(selection["tir_15s"].sum()),
          help=tr("Ce qui s'est vraiment passé, à comparer à la valeur attendue.",
                  "What actually happened, to compare with the expected value."))

onglet_carte, onglet_joueurs, onglet_equipes, onglet_methode = st.tabs(
    [tr("🗺️ Carte des passes", "🗺️ Pass map"), tr("👟 Joueurs", "👟 Players"), tr("🌍 Équipes", "🌍 Teams"),
     tr("🔬 Méthode & validation", "🔬 Method & validation")])


# ---------------------------------------------------------------------------
# Onglet 1 : carte des passes
# ---------------------------------------------------------------------------
COULOIRS = {"gauche": tr("gauche", "left"), "axe": tr("axe", "centre"), "droit": tr("droit", "right")}


def couloir(y: pd.Series) -> pd.Series:
    """Couloir vu par l'équipe qui attaque vers la droite (y = 0 en haut = côté gauche)."""
    return pd.cut(y, [-1, 68 / 3, 2 * 68 / 3, 69], labels=["gauche", "axe", "droit"])


with onglet_carte:
    if affichees.empty:
        st.info(tr("Aucune passe ne correspond à ces filtres. Essayez de baisser le seuil minimum.",
                   "No passes match these filters. Try lowering the minimum threshold."))
    else:
        carte = ajouter_passes(terrain(hauteur=640, barre_couleurs=True), affichees, vmax=VMAX[mesure],
                               mesure=mesure)
        st.plotly_chart(carte, theme=None, config=CONFIG_TERRAIN)
        if mesure == "danger":
            quoi = tr("les plus dangereuses", "most dangerous passes")
        else:
            quoi = tr("qui ajoutent le plus de valeur", "passes adding the most value")
        st.caption(tr(f"Les {len(affichees)} passes {quoi} de la sélection. Toutes les passes sont "
                      "orientées vers la droite (sens d'attaque de l'équipe qui passe). Survolez une flèche pour "
                      "le détail.",
                      f"The {len(affichees)} {quoi} in the selection. All passes are oriented to the right "
                      "(direction of attack of the passing team). Hover over an arrow for details."))

        # --- Lecture rapide ---
        fortes = selection[selection[mesure] >= seuil_fort]
        if mesure == "danger":
            nom_fortes = tr("passes très dangereuses", "very dangerous passes")
        else:
            nom_fortes = tr("passes à forte valeur ajoutée", "high value-added passes")
        titre_l1 = tr(f"D'où partent les {nom_fortes}", f"Where the {nom_fortes} start")
        titre_l2 = tr("Où arrivent-elles", "Where they end")
        st.markdown(tr("#### Lecture rapide", "#### At a glance"))
        l1, l2, l3 = st.columns(3)
        if len(fortes) >= 5:
            part = couloir(fortes["y_debut"]).value_counts(normalize=True)
            l1.markdown(carte_html(
                titre_l1,
                f"<span class='gros-chiffre'>{pct(part.max(), 0)}</span><br><span class='discret'>"
                + tr(f"depuis le couloir {COULOIRS[part.idxmax()]} (gauche {pct(part.get('gauche', 0), 0)} · axe "
                     f"{pct(part.get('axe', 0), 0)} · droit {pct(part.get('droit', 0), 0)})",
                     f"from the {COULOIRS[part.idxmax()]} channel (left {pct(part.get('gauche', 0), 0)} · centre "
                     f"{pct(part.get('axe', 0), 0)} · right {pct(part.get('droit', 0), 0)})")
                + "</span>"),
                unsafe_allow_html=True)
            if mesure == "danger":
                derriere = pct((fortes['derriere_defense'] > 0).mean(), 0)
                l2.markdown(carte_html(
                    titre_l2,
                    f"<span class='gros-chiffre'>{pct(fortes['dans_surface_fin'].mean(), 0)}</span><br>"
                    "<span class='discret'>"
                    + tr(f"dans la surface · {derriere} dans le dos du dernier défenseur visible",
                         f"inside the box · {derriere} in behind the last visible defender")
                    + "</span>"),
                    unsafe_allow_html=True)
            else:
                progression = f"{fortes['progression'].mean():.0f}"
                l2.markdown(carte_html(
                    tr("Comment elles font progresser", "How they move the team forward"),
                    f"<span class='gros-chiffre'>{pct((fortes['adv_elimines'] >= 3).mean(), 0)}</span><br>"
                    "<span class='discret'>"
                    + tr(f"éliminent au moins 3 adversaires · progression moyenne de {progression} m vers le but",
                         f"bypass at least 3 opponents · average progression of {progression} m towards goal")
                    + "</span>"),
                    unsafe_allow_html=True)
        else:
            texte = ("<span class='discret'>" + tr(f"Pas assez de {nom_fortes} dans la sélection.",
                                                   f"Not enough {nom_fortes} in the selection.") + "</span>")
            l1.markdown(carte_html(titre_l1, texte), unsafe_allow_html=True)
            l2.markdown(carte_html(titre_l2, texte), unsafe_allow_html=True)
        meilleurs = selection.groupby(["joueur", "equipe"])[mesure].sum().sort_values(ascending=False).head(3)
        formater = dec if mesure == "danger" else points
        lignes = "".join(f"<li><b>{j}</b> ({e}) · {formater(v)}</li>" for (j, e), v in meilleurs.items())
        if mesure == "danger":
            titre_l3 = tr("Passeurs qui créent le plus de danger", "Passers who create the most danger")
            detail_l3 = tr("somme des probabilités de tir de leurs passes", "sum of the shot probabilities of their passes")
        else:
            titre_l3 = tr("Passeurs qui ajoutent le plus de valeur", "Passers who add the most value")
            detail_l3 = tr("somme des gains de probabilité de leurs passes, en points",
                           "sum of the probability gains of their passes, in points")
        l3.markdown(carte_html(titre_l3, f"<ul>{lignes}</ul><span class='discret'>{detail_l3}</span>"),
                    unsafe_allow_html=True)

        # --- Revoir une passe ---
        st.markdown(tr("#### Revoir une passe", "#### Replay a pass"))
        st.caption(tr("Cliquez sur une ligne pour voir la position des joueurs au moment de la passe.",
                      "Click a row to see the players' positions at the moment of the pass."))
        colonnes = {"joueur": tr("Passeur", "Passer"), "receveur": tr("Receveur", "Receiver"),
                    "equipe": tr("Équipe", "Team"), "adversaire": tr("Adversaire", "Opponent"),
                    "phase_jeu_lib": tr("Phase de jeu", "Phase of play"), "Suite": tr("Suite", "Outcome")}
        tableau = affichees.assign(
            Minute=affichees["minute"].astype(str) + "'",
            Suite=np.where(affichees["but_15s"], tr("⚽ But", "⚽ Goal"),
                           np.where(affichees["tir_15s"], tr("Tir", "Shot"), "—")),
            Danger=affichees["danger"] * 100,
            Valeur=affichees["valeur_ajoutee"] * 100,
        ).rename(columns=colonnes)
        evenement = st.dataframe(
            tableau[["Minute", colonnes["joueur"], colonnes["receveur"], colonnes["equipe"], colonnes["adversaire"],
                     colonnes["phase_jeu_lib"], "Danger", "Valeur", colonnes["Suite"]]],
            hide_index=True, height=290, on_select="rerun", selection_mode="single-row", key="table_passes",
            column_config={
                "Danger": st.column_config.ProgressColumn(
                    "Danger", format=FORMAT_PCT, min_value=0, max_value=VMAX["danger"] * 100,
                    help=tr("Probabilité qu'un tir suive dans les 15 secondes",
                            "Probability that a shot follows within 15 seconds")),
                "Valeur": st.column_config.NumberColumn(
                    tr("Valeur ajoutée", "Value added"), format="%+.1f pts",
                    help=tr("Gain de probabilité de tir par rapport à la situation avant la passe",
                            "Gain in shot probability compared with the situation before the pass")),
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
            st.caption(tr("Zone en pointillés : partie du terrain visible à la caméra. Les joueurs hors champ ne "
                          "sont pas connus (limite des données 360, issues des images TV).",
                          "Dotted area: part of the pitch visible to the camera. Players out of shot are not "
                          "known (a limitation of 360 data, which comes from TV footage)."))
        with d:
            xg = dec(passe['xg_15s'])
            if passe["but_15s"]:
                suite = tr(f"⚽ <b>But</b> dans les 15 secondes (xG du tir : {xg})",
                           f"⚽ <b>Goal</b> within 15 seconds (shot xG: {xg})")
            elif passe["tir_15s"]:
                suite = tr(f"Un <b>tir</b> a suivi (xG : {xg})", f"A <b>shot</b> followed (xG: {xg})")
            else:
                suite = tr("Pas de tir dans les 15 secondes", "No shot within 15 seconds")
            technique = ", ".join(t for t, ok in [(tr("en profondeur", "through ball"), passe["en_profondeur"]),
                                                  (tr("centre", "cross"), passe["centre"]),
                                                  (tr("passe en retrait", "cut-back"), passe["passe_en_retrait"]),
                                                  (tr("renversement", "switch of play"), passe["renversement"])] if ok)
            if passe["derriere_defense"] > 0:
                dos = tr(f"oui, {passe['derriere_defense']:.0f} m derrière le dernier défenseur visible",
                         f"yes, {passe['derriere_defense']:.0f} m behind the last visible defender")
            else:
                dos = tr("non", "no")
            hauteur = HAUTEURS[LANGUE].get(passe["hauteur"], tr("Passe", "Pass"))
            contexte = tr(f"""<ul>
<li>Arrivée à <b>{passe['dist_but_fin']:.0f} m</b> du but, angle de tir de {passe['angle_but_fin']:.0f}°</li>
<li><b>{passe['adv_elimines']}</b> adversaire(s) éliminé(s)</li>
<li><b>{passe['defenseurs_restants']}</b> défenseur(s) encore entre le ballon et le but</li>
<li>Dans le dos de la défense : {dos}</li>
<li>Adversaire le plus proche du receveur : {dec(passe['dist_adv_fin'], 1)} m</li>
<li>{hauteur}{', ' + technique if technique else ''} · {passe['phase_jeu_lib'].lower()}</li></ul>""",
                          f"""<ul>
<li>Ends <b>{passe['dist_but_fin']:.0f} m</b> from goal, shooting angle of {passe['angle_but_fin']:.0f}°</li>
<li><b>{passe['adv_elimines']}</b> opponent(s) bypassed</li>
<li><b>{passe['defenseurs_restants']}</b> defender(s) still between the ball and the goal</li>
<li>In behind the defence: {dos}</li>
<li>Nearest opponent to the receiver: {dec(passe['dist_adv_fin'], 1)} m</li>
<li>{hauteur}{', ' + technique if technique else ''} · {passe['phase_jeu_lib'].lower()}</li></ul>""")
            st.markdown(
                carte_html(f"{passe['joueur']} → {passe['receveur']}",
                           f"<span class='discret'>{passe['equipe']} vs {passe['adversaire']} · "
                           f"{passe['minute']}'</span><br><br><span class='gros-chiffre'>{pct(passe['danger'])}"
                           "</span><br><span class='discret'>"
                           + tr(f"de chances de tir dans les 15 s, soit <b>{dec(passe['danger'] / taux_base, 1)}×</b> "
                                "la moyenne d'une passe",
                                f"chance of a shot within 15 s, <b>{dec(passe['danger'] / taux_base, 1)}×</b> "
                                "the average pass")
                           + "</span>")
                + carte_html(tr("Valeur ajoutée", "Value added"),
                             f"<span class='gros-chiffre'>{points(passe['valeur_ajoutee'])}</span><br>"
                             "<span class='discret'>"
                             + tr(f"la probabilité de tir passe de <b>{pct(passe['valeur_avant'])}</b> avant la passe "
                                  f"à <b>{pct(passe['danger'])}</b> après",
                                  f"shot probability goes from <b>{pct(passe['valeur_avant'])}</b> before the pass "
                                  f"to <b>{pct(passe['danger'])}</b> after")
                             + "</span>")
                + carte_html(tr("Le contexte de la passe", "Context of the pass"), contexte)
                + carte_html(tr("Suite de l'action", "What happened next"), suite),
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Onglet 2 : joueurs
# ---------------------------------------------------------------------------
with onglet_joueurs:
    st.markdown(tr("#### Qui crée le plus de danger par la passe ?", "#### Who creates the most danger with their passing?"))
    st.caption(tr(
        "Calculé sur la sélection de match, d'équipe et de type de passe (toutes périodes). "
        "Le « danger créé » additionne les probabilités de tir de toutes les passes d'un joueur : "
        "c'est le nombre de ses passes qu'on pouvait s'attendre à voir suivies d'un tir. La « valeur "
        "ajoutée » additionne, en points, ce que chaque passe a fait gagner (ou perdre) par rapport à la situation "
        "de départ : elle met en avant les joueurs qui font progresser l'équipe, même loin du but. Une valeur "
        "négative n'est pas un défaut : faire tourner le ballon en sécurité (Rodri, Gavi) dégrade "
        "légèrement la situation immédiate mais peut servir le jeu de possession. "
        "Les joueurs des équipes qui ont beaucoup le ballon (Espagne, Allemagne) sont avantagés par le volume.",
        "Calculated on the selected match, team and pass type (all periods). "
        "“Danger created” adds up the shot probabilities of all of a player's passes: "
        "it is the number of his passes that could be expected to be followed by a shot. “Value "
        "added” adds up, in points, what each pass gained (or lost) compared with the starting situation: "
        "it highlights players who move the team forward, even far from goal. A negative value "
        "is not a flaw: circulating the ball safely (Rodri, Gavi) slightly worsens the immediate situation "
        "but can serve a possession game. "
        "Players from teams with a lot of the ball (Spain, Germany) benefit from the volume."))

    mins = minutes if match_id is None else minutes[minutes["match_id"] == match_id]
    if equipe != TOUTES_EQUIPES:
        mins = mins[mins["equipe"] == equipe]
    mins = mins.groupby("joueur_id", as_index=False)["minutes"].sum()

    agr = (base.groupby(["joueur_id", "joueur", "equipe"], as_index=False)
           .agg(passes=("danger", "size"), danger=("danger", "sum"), valeur=("valeur_ajoutee", "sum"),
                tres_dangereuses=("danger", lambda s: int((s >= seuil_top).sum())),
                meilleure=("danger", "max"), tirs=("tir_15s", "sum"))
           .merge(mins, on="joueur_id", how="left"))
    agr["minutes"] = agr["minutes"].fillna(0)
    agr["valeur"] *= 100      # valeur ajoutée en points de probabilité, comme pour une passe seule
    agr["meilleure"] *= 100   # en %
    agr["danger_90"] = agr["danger"] / agr["minutes"].clip(lower=1) * 90
    agr["valeur_90"] = agr["valeur"] / agr["minutes"].clip(lower=1) * 90

    m1, m2 = st.columns(2)
    min_minutes = m1.slider(tr("Minutes jouées minimum", "Minimum minutes played"), 0, 600 if match_id is None else 120,
                            270 if match_id is None else 30, step=10)
    criteres = {"danger_90": tr("Danger créé / 90 min", "Danger created / 90 min"),
                "danger": tr("Danger créé (total)", "Danger created (total)"),
                "valeur_90": tr("Valeur ajoutée / 90 min", "Value added / 90 min"),
                "valeur": tr("Valeur ajoutée (total)", "Value added (total)"),
                "tres_dangereuses": tr("Passes très dangereuses", "Very dangerous passes")}
    cle = m2.radio(tr("Classer par", "Rank by"), list(criteres), format_func=criteres.get, horizontal=True,
                   index=0 if mesure == "danger" else 2)
    classement = agr[agr["minutes"] >= min_minutes].sort_values(cle, ascending=False)

    if classement.empty:
        st.info(tr("Aucun joueur ne correspond. Baissez le nombre de minutes minimum.",
                   "No players match. Lower the minimum number of minutes."))
    else:
        top = classement.head(15).iloc[::-1]
        en_points = cle in ("valeur", "valeur_90")
        if en_points:
            textes, survol = [points(v / 100) for v in top[cle]], "%{x:+.1f} pts"
        elif cle == "tres_dangereuses":
            textes, survol = [f"{v:.0f}" for v in top[cle]], "%{x:.0f}"
        else:
            textes, survol = [dec(v) for v in top[cle]], "%{x:.2f}"
        fig = go.Figure(go.Bar(
            x=top[cle], y=top["joueur"] + "  ·  " + top["equipe"], orientation="h",
            marker=dict(color=top[cle], colorscale=ECHELLE_BARRES, line=dict(width=0)),
            text=textes, textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>" + survol + "<extra></extra>",
        ))
        fig.update_layout(height=540, margin=dict(l=8, r=55, t=10, b=40), **FOND_GRAPHIQUE,
                          xaxis=dict(gridcolor=GRILLE, zeroline=False,
                                     title=criteres[cle] + (tr(" (points de probabilité)", " (probability points)")
                                                            if en_points else "")),
                          yaxis=dict(showgrid=False, automargin=True))
        afficher(fig)

        noms = {"joueur": tr("Joueur", "Player"), "equipe": tr("Équipe", "Team"), "minutes": "Minutes",
                "passes": "Passes", "danger": tr("Danger créé", "Danger created"), "danger_90": "Danger / 90",
                "valeur": tr("Valeur ajoutée", "Value added"), "valeur_90": tr("Valeur / 90", "Value / 90"),
                "tres_dangereuses": tr("Très dangereuses", "Very dangerous"),
                "meilleure": tr("Meilleure passe", "Best pass"), "tirs": tr("Suivies d'un tir", "Followed by a shot")}
        st.dataframe(
            classement.rename(columns=noms),
            hide_index=True, height=380,
            column_order=[noms[c] for c in ("joueur", "equipe", "minutes", "passes", "danger", "danger_90", "valeur",
                                            "valeur_90", "tres_dangereuses", "tirs", "meilleure")],
            column_config={
                noms["minutes"]: st.column_config.NumberColumn(format="%d"),
                noms["danger"]: st.column_config.NumberColumn(
                    format="%.2f", help=tr("Somme des probabilités de tir de ses passes",
                                           "Sum of the shot probabilities of his passes")),
                noms["danger_90"]: st.column_config.ProgressColumn(
                    format="%.2f", min_value=0, max_value=float(max(classement["danger_90"].max(), 0.01))),
                noms["valeur"]: st.column_config.NumberColumn(
                    format="%+.1f pts", help=tr("Somme des gains de probabilité de tir apportés par ses passes",
                                                "Sum of the gains in shot probability brought by his passes")),
                noms["valeur_90"]: st.column_config.NumberColumn(format="%+.1f pts"),
                noms["meilleure"]: st.column_config.NumberColumn(
                    format=FORMAT_PCT, help=tr("Probabilité de tir après sa passe la plus dangereuse",
                                               "Shot probability after his most dangerous pass")),
            },
        )


# ---------------------------------------------------------------------------
# Onglet 3 : équipes
# ---------------------------------------------------------------------------
with onglet_equipes:
    st.markdown(tr("#### Danger créé et concédé par match", "#### Danger created and conceded per match"))
    st.caption(tr("Sur tout le tournoi, ", "Across the whole tournament, ")
               + (tr("tous types de passes.", "all pass types.") if cpa
                  else tr("jeu courant uniquement.", "open play only.")))
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
        hovertemplate=tr("<b>%{text}</b><br>Créé : %{x:.2f} / match<br>Concédé : %{y:.2f} / match"
                         "<br>%{customdata[0]} matchs<extra></extra>",
                         "<b>%{text}</b><br>Created: %{x:.2f} / match<br>Conceded: %{y:.2f} / match"
                         "<br>%{customdata[0]} matches<extra></extra>"),
    ))
    fig.add_vline(x=eq["cree"].median(), line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
    fig.add_hline(y=eq["concede"].median(), line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
    fig.update_layout(height=580, margin=dict(l=10, r=10, t=20, b=10), **FOND_GRAPHIQUE,
                      xaxis=dict(title=tr("Danger créé par match (somme des probabilités)",
                                          "Danger created per match (sum of probabilities)"), gridcolor=GRILLE,
                                 zeroline=False, automargin=True),
                      yaxis=dict(title=tr("Danger concédé par match", "Danger conceded per match"), gridcolor=GRILLE,
                                 zeroline=False, autorange="reversed", automargin=True))
    afficher(fig)
    st.caption(tr("Axe vertical inversé : en haut à droite, les équipes dangereuses et solides ; en bas à gauche, "
                  "celles qui créent peu et concèdent beaucoup. Pointillés : médianes. "
                  "Taille des points = nombre de matchs joués.",
                  "Vertical axis reversed: top right, teams that are dangerous and solid; bottom left, "
                  "those that create little and concede a lot. Dotted lines: medians. "
                  "Dot size = number of matches played."))

    st.markdown(tr("#### Profil d'une équipe", "#### Team profile"))
    equipes_triees = sorted(eq.index)
    defaut = equipe if equipe in equipes_triees else tr("Argentine", "Argentina")
    choix = st.selectbox(tr("Équipe à analyser", "Team to analyse"), equipes_triees, index=equipes_triees.index(defaut))
    attaque = pe[(pe["equipe"] == choix) & (pe["danger"] >= seuil_top)]
    defense = pe[(pe["adversaire"] == choix) & (pe["danger"] >= seuil_top)]
    a, b = st.columns(2)
    with a:
        st.plotly_chart(carte_chaleur(attaque["x_debut"], attaque["y_debut"], poids=attaque["danger"],
                                      titre=tr("D'où partent ses passes très dangereuses",
                                               "Where its very dangerous passes start"), hauteur=430),
                        theme=None, config=CONFIG_SIMPLE)
        st.caption(tr(f"{len(attaque)} passes très dangereuses de : {choix} · attaque vers la droite.",
                      f"{len(attaque)} very dangerous passes by {choix} · attacking to the right."))
    with b:
        st.plotly_chart(carte_chaleur(defense["x_fin"], defense["y_fin"], poids=defense["danger"],
                                      titre=tr("Où ses adversaires font arriver leurs passes très dangereuses",
                                               "Where its opponents' very dangerous passes end"),
                                      couleur="255,107,107", hauteur=430),
                        theme=None, config=CONFIG_SIMPLE)
        st.caption(tr(f"{len(defense)} passes très dangereuses concédées · le but de : {choix} est à droite.",
                      f"{len(defense)} very dangerous passes conceded · {choix}'s goal is on the right."))


# ---------------------------------------------------------------------------
# Onglet 4 : méthode et validation
# ---------------------------------------------------------------------------
with onglet_methode:
    res, res3 = rapport["resultats"], rapport["resultats_dernier_tiers"]
    retenu = rapport["modele_retenu"]
    gain360, gain_ancien = rapport["gain_contexte_360"], rapport["gain_vs_ancienne_formule"]
    ancienne = "Ancienne formule (poids à la main)"

    st.markdown(tr("#### La question", "#### The question"))
    st.markdown(tr(
        "Toutes les passes ne se valent pas : une passe latérale entre défenseurs ne crée rien, une passe qui "
        "élimine trois adversaires et arrive seule face au but change le match. L'objectif est de **mesurer "
        "ce danger de façon objective**, puis de vérifier que la mesure tient la route.",
        "Not all passes are equal: a sideways pass between defenders creates nothing, while a pass that "
        "bypasses three opponents and finds a teammate alone in front of goal changes the match. The aim is to "
        "**measure this danger objectively**, then check that the measure holds up."))

    st.markdown(tr("#### Les données", "#### The data"))
    st.markdown(tr(
        f"- **StatsBomb Open Data**, Coupe du monde 2022 : {rapport['n_matchs']} matchs, chaque action annotée.\n"
        "- **StatsBomb 360** : pour chaque action, la position de tous les joueurs visibles à l'image. "
        "C'est ce qui permet de savoir combien d'adversaires une passe élimine ou quel espace a le receveur.\n"
        f"- {nombre(rapport['n_passes'])} passes réussies retenues, soit {pct(rapport['couverture_360'], 0)} "
        "des passes réussies (les autres n'ont pas d'image 360, par exemple pendant un ralenti).",
        f"- **StatsBomb Open Data**, 2022 World Cup: {rapport['n_matchs']} matches, every action annotated.\n"
        "- **StatsBomb 360**: for each action, the position of every player visible in the footage. "
        "This is what tells us how many opponents a pass bypasses or how much space the receiver has.\n"
        f"- {nombre(rapport['n_passes'])} completed passes kept, i.e. {pct(rapport['couverture_360'], 0)} "
        "of completed passes (the others have no 360 frame, for example during a replay)."))

    st.markdown(tr("#### Ce que le modèle prédit", "#### What the model predicts"))
    st.markdown(tr(
        f"La cible est simple et vérifiable : **la même équipe tire-t-elle dans les 15 secondes, sans perdre "
        f"le ballon ?** C'est le cas pour {pct(taux_base)} des passes. Le score de danger d'une passe est la "
        "probabilité estimée de cet événement. Les variables décrivent uniquement ce qu'on sait au moment de "
        "la passe, jamais la suite de l'action.",
        f"The target is simple and verifiable: **does the same team shoot within 15 seconds, without losing "
        f"the ball?** This is the case for {pct(taux_base)} of passes. A pass's danger score is the "
        "estimated probability of this event. The features only describe what is known at the moment of "
        "the pass, never what happens next."))

    st.markdown("#### Validation")
    st.markdown(tr(
        "Le modèle est évalué en **validation croisée par match** : il est entraîné sur 4/5 des matchs et "
        "noté sur les matchs restants, cinq fois de suite. Chaque score affiché dans l'application vient donc "
        "d'un modèle qui n'a jamais vu le match en question. L'**AUC** mesure la capacité à bien classer les "
        "passes (0,5 = hasard, 1 = parfait).",
        "The model is evaluated with **match-grouped cross-validation**: it is trained on 4/5 of the matches and "
        "scored on the remaining matches, five times over. Every score shown in the app therefore comes "
        "from a model that has never seen the match in question. **AUC** measures how well the model ranks "
        "passes (0.5 = random, 1 = perfect)."))

    ordre = [ancienne, "Distance au but seule", "Position seule (sans 360)", "Gradient boosting",
             "Régression logistique"]
    noms = [modele(n) + " ✓" if n == retenu else modele(n) for n in ordre]
    fig = go.Figure()
    for jeu, nom_jeu, couleur in [(res, tr("Toutes les passes", "All passes"), "#3dd1ff"),
                                  (res3, tr("Passes vers le dernier tiers", "Passes into the final third"), "#6bd3a8")]:
        fig.add_trace(go.Bar(y=noms, x=[jeu[n]["auc"] for n in ordre], orientation="h", name=nom_jeu,
                             marker_color=couleur, text=[dec(jeu[n]["auc"], 3) for n in ordre],
                             textposition="outside", cliponaxis=False))
    fig.update_layout(barmode="group", height=430, margin=dict(l=8, r=50, t=10, b=10), **FOND_GRAPHIQUE,
                      xaxis=dict(range=[0.5, 0.9], title=tr("AUC (validation par match)", "AUC (match-grouped validation)"),
                                 gridcolor=GRILLE),
                      yaxis=dict(autorange="reversed", automargin=True), legend=dict(orientation="h", y=1.1, x=0))
    afficher(fig)

    st.markdown(carte_html(tr("Ce qu'il faut retenir", "Key takeaways"), tr(f"""<ul>
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
facile à expliquer.</li></ul>""", f"""<ul>
<li><b>The first version of the project</b> (hand-picked weights) could barely tell dangerous passes apart
in the final third: AUC of {dec(res3[ancienne]['auc'])}, against {dec(res3[retenu]['auc'])} for
the model. Across all passes, the gain is +{dec(gain_ancien['gain'], 3)} (95% confidence interval:
{dec(gain_ancien['ic95'][0], 3)} to {dec(gain_ancien['ic95'][1], 3)}).</li>
<li><b>Where the pass ends explains most of the danger</b>: distance to goal alone already reaches
{dec(res['Distance au but seule']['auc'], 3)}.</li>
<li><b>360 context brings a real but modest gain</b>: +{dec(gain360['gain'], 3)} AUC compared with
a model that only knows the position (95% CI: {dec(gain360['ic95'][0], 3)} to {dec(gain360['ic95'][1], 3)}).
This is expected: TV footage does not show every player, and what happens next depends on many other
things than the pass.</li>
<li><b>{modele(retenu)}</b> is the chosen model: as accurate as gradient boosting, better calibrated
(log loss {dec(res[retenu]['log_loss'], 4)} against {dec(res['Gradient boosting']['log_loss'], 4)}) and
easier to explain.</li></ul>""")), unsafe_allow_html=True)

    g, d = st.columns(2)
    with g:
        st.markdown(tr("##### Les probabilités sont-elles fiables ?", "##### Are the probabilities reliable?"))
        deciles = pd.DataFrame(rapport["deciles"])
        lim = float(max(deciles["proba_moyenne"].max(), deciles["taux_observe"].max())) * 110
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", hoverinfo="skip",
                                 line=dict(color="rgba(255,255,255,0.25)", dash="dot")))
        fig.add_trace(go.Scatter(x=deciles["proba_moyenne"] * 100, y=deciles["taux_observe"] * 100,
                                 mode="lines+markers", line=dict(color="#3dd1ff", width=2.5),
                                 marker=dict(size=9, color="#3dd1ff"), customdata=deciles[["decile", "passes"]],
                                 hovertemplate=tr("Tranche %{customdata[0]} (%{customdata[1]} passes)<br>"
                                                  "Prédit : %{x:.1f} %<br>Observé : %{y:.1f} %<extra></extra>",
                                                  "Decile %{customdata[0]} (%{customdata[1]} passes)<br>"
                                                  "Predicted: %{x:.1f}%<br>Observed: %{y:.1f}%<extra></extra>")))
        fig.update_layout(height=380, margin=dict(l=8, r=8, t=10, b=10), **FOND_GRAPHIQUE, showlegend=False,
                          xaxis=dict(title=tr("Probabilité prédite (%)", "Predicted probability (%)"), gridcolor=GRILLE),
                          yaxis=dict(title=tr("Fréquence réelle de tir (%)", "Actual shot frequency (%)"),
                                     gridcolor=GRILLE))
        afficher(fig)
        dernier = deciles.iloc[-1]
        annonce, observe = f"{dernier['proba_moyenne'] * 100:.0f}", f"{dernier['taux_observe'] * 100:.0f}"
        st.caption(tr(f"Passes regroupées en 10 tranches de danger. Les points suivent la diagonale : pour les "
                      f"10 % de passes les plus dangereuses, le modèle annonce {annonce} % en moyenne, et un tir suit "
                      f"réellement dans {observe} % des cas.",
                      f"Passes grouped into 10 danger deciles. The points follow the diagonal: for the "
                      f"10% most dangerous passes, the model predicts {annonce}% on average, and a shot actually "
                      f"follows in {observe}% of cases."))
    with d:
        st.markdown(tr("##### Quelles informations comptent le plus ?", "##### Which information matters most?"))
        imp = pd.Series(rapport["importance"]).clip(lower=0).sort_values()
        fig = go.Figure(go.Bar(x=imp.values * 100, y=[libelle(n, VARIABLES_EN) for n in imp.index], orientation="h",
                               marker=dict(color=imp.values, colorscale=ECHELLE_BARRES),
                               text=[dec(v, 2) for v in imp.values * 100], textposition="outside",
                               cliponaxis=False,
                               hovertemplate=tr("%{y}<br>Perte d'AUC : %{x:.2f} point(s)<extra></extra>",
                                                "%{y}<br>AUC loss: %{x:.2f} point(s)<extra></extra>")))
        fig.update_layout(height=420, margin=dict(l=8, r=45, t=10, b=10), **FOND_GRAPHIQUE,
                          xaxis=dict(title=tr("Perte d'AUC si on brouille l'information (points)",
                                              "AUC loss when the information is shuffled (points)"),
                                     gridcolor=GRILLE), yaxis=dict(automargin=True))
        afficher(fig)
        st.caption(tr("Importance par permutation : on mélange une famille de variables et "
                      "on mesure combien le modèle se dégrade.",
                      "Permutation importance: one family of features is shuffled and "
                      "we measure how much the model deteriorates."))

    st.markdown(tr("##### Où une passe devient-elle dangereuse ?", "##### Where does a pass become dangerous?"))
    jeu_courant = passes[passes["type_passe"] == "Jeu courant"]
    st.plotly_chart(carte_chaleur(jeu_courant["x_fin"], jeu_courant["y_fin"], moyenne=True,
                                  valeurs=jeu_courant["danger"], nx=21, ny=14, hauteur=520,
                                  suffixe=tr(" %", "%")),
                    theme=None, config=CONFIG_SIMPLE)
    st.caption(tr("Danger moyen selon la zone d'arrivée de la passe (jeu courant, zones d'au moins 30 passes).",
                  "Average danger by the zone where the pass ends (open play, zones with at least 30 passes)."))

    st.markdown(tr("#### La valeur ajoutée : récompenser aussi la progression",
                   "#### Value added: rewarding progression too"))
    avant, va = rapport["valeur_avant"], rapport["valeur_ajoutee"]
    auc_avant = dec(avant['resultats'][avant['modele_retenu']]['auc'], 3)
    st.markdown(tr(
        "Le danger mesure la **proximité d'un tir** : une passe qui casse une ligne au milieu de terrain reste "
        "loin du but, donc peu dangereuse, même si elle fait beaucoup progresser l'équipe. Pour la récompenser, "
        "un second modèle estime la probabilité de tir **avant la passe**, à partir de la seule situation du "
        "passeur (position du ballon, pression, défenseurs entre lui et le but, phase de jeu), sans connaître la "
        "destination. La **valeur ajoutée** est la différence :\n\n"
        "> valeur ajoutée = danger après la passe − probabilité de tir avant la passe\n\n"
        f"Elle s'exprime en points de probabilité et vaut zéro en moyenne ({dec(va['moyenne'], 2)} pt) : une "
        "passe positive améliore la situation, une passe négative (une passe en retrait, par exemple) la dégrade. "
        f"Le modèle « avant la passe » ({avant['modele_retenu'].lower()}) atteint une AUC de "
        f"{auc_avant} en validation par match.",
        "Danger measures **how close a shot is**: a line-breaking pass in midfield is still "
        "far from goal, so not very dangerous, even if it moves the team forward a lot. To reward it, "
        "a second model estimates the shot probability **before the pass**, from the passer's situation alone "
        "(ball position, pressure, defenders between him and the goal, phase of play), without knowing the "
        "destination. **Value added** is the difference:\n\n"
        "> value added = danger after the pass − shot probability before the pass\n\n"
        f"It is expressed in probability points and is zero on average ({dec(va['moyenne'], 2)} pt): a "
        "positive pass improves the situation, a negative pass (a back pass, for example) worsens it. "
        f"The “before the pass” model ({modele(avant['modele_retenu']).lower()}) reaches an AUC of "
        f"{auc_avant} in match-grouped validation."))

    par_zone = pd.DataFrame(va["par_zone_et_elimines"])
    fig = go.Figure()
    for niveau, couleur in [("0", "#2d7fb3"), ("1-2", "#3dd1ff"), ("3+", "#6bd3a8")]:
        sous = par_zone[par_zone["elimines"] == niveau]
        fig.add_trace(go.Bar(x=[libelle(z, ZONES_EN) for z in sous["zone"]], y=sous["valeur_ajoutee"],
                             name=tr(f"{niveau} adversaire(s) éliminé(s)", f"{niveau} opponent(s) bypassed"),
                             marker_color=couleur, customdata=sous[["passes", "danger"]],
                             text=[dec(v, 1) if v < 0 else "+" + dec(v, 1) for v in sous["valeur_ajoutee"]],
                             textposition="outside", cliponaxis=False,
                             hovertemplate=tr("%{x}<br>Valeur ajoutée moyenne : %{y:+.2f} pts<br>"
                                              "Danger moyen : %{customdata[1]:.1f} %<br>%{customdata[0]} passes"
                                              "<extra></extra>",
                                              "%{x}<br>Average value added: %{y:+.2f} pts<br>"
                                              "Average danger: %{customdata[1]:.1f}%<br>%{customdata[0]} passes"
                                              "<extra></extra>")))
    fig.update_layout(barmode="group", height=420, margin=dict(l=8, r=8, t=10, b=10), **FOND_GRAPHIQUE,
                      xaxis=dict(title=tr("Zone d'arrivée de la passe", "Zone where the pass ends")),
                      yaxis=dict(title=tr("Valeur ajoutée moyenne (points)", "Average value added (points)"),
                                 gridcolor=GRILLE, zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
                      legend=dict(orientation="h", y=1.1, x=0))
    afficher(fig)
    correlation = va["correlation_adv_elimines"]
    st.caption(tr("Valeur ajoutée moyenne selon la zone d'arrivée et le nombre d'adversaires éliminés (jeu courant). "
                  f"Corrélation (Spearman) avec le nombre d'adversaires éliminés : {dec(correlation['danger'], 2)} "
                  f"pour le danger, {dec(correlation['valeur_ajoutee'], 2)} pour la valeur ajoutée. Limite : dans sa "
                  "propre moitié, même une passe qui élimine 3 adversaires ajoute peu, car la probabilité de tir y "
                  "reste très faible avant comme après la passe.",
                  "Average value added by end zone and number of opponents bypassed (open play). "
                  f"Correlation (Spearman) with the number of opponents bypassed: {dec(correlation['danger'], 2)} "
                  f"for danger, {dec(correlation['valeur_ajoutee'], 2)} for value added. Limitation: in a team's "
                  "own half, even a pass that bypasses 3 opponents adds little, because the shot probability there "
                  "remains very low both before and after the pass."))

    st.markdown(tr("#### Limites", "#### Limitations"))
    st.markdown(tr(
        "- **Joueurs hors champ** : les données 360 viennent des images TV. Un défenseur hors cadre n'est pas "
        "compté, d'où la variable « adversaires visibles » qui sert de contrôle.\n"
        "- **Une photo, pas un film** : les positions sont celles du moment de la passe, pas de la réception.\n"
        "- **Un seul tournoi** : 64 matchs. Le modèle reste à tester sur d'autres compétitions (Euro 2024, "
        "Ligue 1) avant d'en tirer des conclusions générales.\n"
        "- **Seules les passes réussies** sont notées : le risque d'une passe tentée n'est pas mesuré.",
        "- **Players out of shot**: 360 data comes from TV footage. A defender out of frame is not "
        "counted, hence the “visible opponents” feature, which serves as a control.\n"
        "- **A snapshot, not a film**: positions are those at the moment of the pass, not of the reception.\n"
        "- **A single tournament**: 64 matches. The model still needs testing on other competitions (Euro 2024, "
        "Ligue 1) before drawing general conclusions.\n"
        "- **Only completed passes** are scored: the risk of an attempted pass is not measured."))

st.markdown(tr(
    "<div class='pied'>Projet réalisé par <b>Mathis Maitrallin</b>, Data Scientist. "
    "Données : <a href='https://github.com/statsbomb/open-data' target='_blank'>StatsBomb Open Data</a> "
    "(Coupe du monde 2022, 360). Python · pandas · scikit-learn · Plotly · Streamlit.</div>",
    "<div class='pied'>Project by <b>Mathis Maitrallin</b>, Data Scientist. "
    "Data: <a href='https://github.com/statsbomb/open-data' target='_blank'>StatsBomb Open Data</a> "
    "(2022 World Cup, 360). Python · pandas · scikit-learn · Plotly · Streamlit.</div>"),
    unsafe_allow_html=True)
