"""
Dessin du terrain et des passes avec Plotly (interactif : survol, zoom).

Repère : mètres, 105 x 68, attaque vers la droite, y = 0 en haut (vue "télé").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

L, H = 105.0, 68.0

COULEURS = {
    "fond": "rgba(0,0,0,0)",
    "pelouse": "#0d1a17",
    "lignes": "rgba(230,238,248,0.30)",
    "texte": "#c9d4e3",
    "attaque": "#3dd1ff",
    "defense": "#ff6b6b",
    "neutre": "#97a0b3",
}

ECHELLE_DANGER = [
    [0.0, "#2a4258"],
    [0.3, "#2d7fb3"],
    [0.55, "#3dd1ff"],
    [0.8, "#8fe3c0"],
    [1.0, "#ffd166"],
]


def couleur_danger(valeurs, vmax: float) -> list[str]:
    v = np.clip(np.asarray(valeurs, dtype=float) / max(vmax, 1e-9), 0, 1)
    return sample_colorscale(ECHELLE_DANGER, list(v))


def _arc(cx, cy, r, theta1, theta2, n=40) -> str:
    t = np.radians(np.linspace(theta1, theta2, n))
    pts = [f"{cx + r * np.cos(a):.2f},{cy + r * np.sin(a):.2f}" for a in t]
    return "M " + " L ".join(pts)


def terrain(hauteur: int = 560, titre: str | None = None, barre_couleurs: bool = False) -> go.Figure:
    """Figure Plotly vide avec un terrain dessiné (barre_couleurs : réserve la place d'une échelle en bas)."""
    ligne = dict(color=COULEURS["lignes"], width=1.6)
    formes = [
        dict(type="rect", x0=0, y0=0, x1=L, y1=H, line=ligne, fillcolor=COULEURS["pelouse"], layer="below"),
        dict(type="line", x0=L / 2, y0=0, x1=L / 2, y1=H, line=ligne),
        dict(type="circle", x0=L / 2 - 9.15, y0=H / 2 - 9.15, x1=L / 2 + 9.15, y1=H / 2 + 9.15, line=ligne),
        dict(type="circle", x0=L / 2 - 0.4, y0=H / 2 - 0.4, x1=L / 2 + 0.4, y1=H / 2 + 0.4,
             line=ligne, fillcolor=COULEURS["lignes"]),
    ]
    for cote in (0, 1):
        x_ligne = L if cote else 0
        s = -1 if cote else 1  # sens vers l'intérieur du terrain
        formes += [
            dict(type="rect", x0=x_ligne, y0=H / 2 - 20.16, x1=x_ligne + s * 16.5, y1=H / 2 + 20.16, line=ligne),
            dict(type="rect", x0=x_ligne, y0=H / 2 - 9.16, x1=x_ligne + s * 5.5, y1=H / 2 + 9.16, line=ligne),
            dict(type="rect", x0=x_ligne, y0=H / 2 - 3.66, x1=x_ligne - s * 1.8, y1=H / 2 + 3.66, line=ligne),
            dict(type="circle", x0=x_ligne + s * 11 - 0.4, y0=H / 2 - 0.4, x1=x_ligne + s * 11 + 0.4,
                 y1=H / 2 + 0.4, line=ligne, fillcolor=COULEURS["lignes"]),
        ]
        # arc de cercle de la surface : portion du cercle (r = 9.15) hors de la surface
        demi = np.degrees(np.arccos(5.5 / 9.15))
        centre = 0 if not cote else 180
        formes.append(dict(type="path", path=_arc(x_ligne + s * 11, H / 2, 9.15, centre - demi, centre + demi),
                           line=ligne))

    fig = go.Figure()
    fig.update_layout(
        shapes=formes,
        height=hauteur,
        margin=dict(l=8, r=8, t=40 if titre else 8, b=70 if barre_couleurs else 8),
        paper_bgcolor=COULEURS["fond"],
        plot_bgcolor=COULEURS["fond"],
        xaxis=dict(range=[-3, L + 3], visible=False, constrain="domain"),
        yaxis=dict(range=[H + 3, -3], visible=False, scaleanchor="x", scaleratio=1, constrain="domain"),
        showlegend=False,
        hoverlabel=dict(bgcolor="#0f1620", bordercolor="#3dd1ff", font=dict(color="#e6eef8", size=13),
                        align="left"),
        font=dict(color=COULEURS["texte"], family="Inter, system-ui, sans-serif"),
        title=dict(text=titre, x=0.01, font=dict(size=15)) if titre else None,
        dragmode="pan",
    )
    fig.add_annotation(x=L / 2, y=-1.2, text="Sens d'attaque  →", showarrow=False,
                       font=dict(size=11, color=COULEURS["neutre"]), yanchor="bottom")
    return fig


MESURES = {
    "danger": {"titre": "Danger (probabilité de tir)", "suffixe": " %"},
    "valeur_ajoutee": {"titre": "Valeur ajoutée (points de probabilité)", "suffixe": " pts"},
}


def ajouter_passes(fig: go.Figure, passes: pd.DataFrame, vmax: float | None = None,
                   n_classes: int = 8, colorbar: bool = True, mesure: str = "danger") -> go.Figure:
    """
    Ajoute des flèches colorées selon une mesure ("danger" ou "valeur_ajoutee"), avec un survol détaillé.
    Les valeurs négatives (passes qui dégradent la situation) prennent la couleur la plus froide.
    """
    if passes.empty:
        return fig
    vmax = vmax or float(passes[mesure].max())
    p = passes.sort_values(mesure)  # les plus fortes dessinées au-dessus
    valeurs = p[mesure].clip(lower=0)
    classes = np.minimum((valeurs / vmax * n_classes).astype(int), n_classes - 1)

    for c in range(n_classes):
        sous = p[classes == c]
        if sous.empty:
            continue
        couleur = couleur_danger([(c + 0.5) / n_classes * vmax], vmax)[0]
        xs = np.column_stack([sous["x_debut"], sous["x_fin"], np.full(len(sous), np.nan)]).ravel()
        ys = np.column_stack([sous["y_debut"], sous["y_fin"], np.full(len(sous), np.nan)]).ravel()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                 line=dict(color=couleur, width=1.4 + 2.2 * c / n_classes),
                                 opacity=0.55 + 0.45 * c / n_classes))

    # points de départ
    fig.add_trace(go.Scatter(x=p["x_debut"], y=p["y_debut"], mode="markers", hoverinfo="skip",
                             marker=dict(size=4, color=couleur_danger(valeurs, vmax), opacity=0.8)))

    # têtes de flèche + survol (angle Plotly : 0 = vers le haut de l'écran, sens horaire ;
    # l'axe y est inversé, donc un y croissant descend à l'écran)
    angles = np.degrees(np.arctan2(p["x_fin"] - p["x_debut"], -(p["y_fin"] - p["y_debut"])))
    suite = np.where(p["but_15s"], "⚽ but", np.where(p["tir_15s"], "tir", "pas de tir"))
    custom = np.column_stack([
        p["joueur"], p["receveur"], p["equipe"], p["adversaire"],
        p["minute"].astype(str) + "'", (p["danger"] * 100).round(1),
        (p["valeur_avant"] * 100).round(1), (p["valeur_ajoutee"] * 100).map(lambda v: f"{v:+.1f}".replace(".", ",")),
        p["dist_but_fin"].round(0).astype(int), p["adv_elimines"], p["defenseurs_restants"], suite,
    ])
    graduations = np.linspace(0, vmax, 5)
    infos = MESURES[mesure]
    fig.add_trace(go.Scatter(
        x=p["x_fin"], y=p["y_fin"], mode="markers", customdata=custom,
        marker=dict(symbol="arrow", size=11, angle=angles, color=valeurs, cmin=0, cmax=vmax,
                    colorscale=ECHELLE_DANGER, line=dict(width=0),
                    showscale=colorbar,
                    colorbar=dict(title=dict(text=infos["titre"], side="top"),
                                  orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top",
                                  tickvals=graduations,
                                  ticktext=[f"{v * 100:.0f}{infos['suffixe']}" for v in graduations],
                                  thickness=10, len=0.5, outlinewidth=0) if colorbar else None),
        hovertemplate=(
            "<b>%{customdata[0]}</b> → %{customdata[1]}<br>"
            "%{customdata[2]} vs %{customdata[3]} · %{customdata[4]}<br>"
            "<b>Danger : %{customdata[5]} %</b> de chances de tir dans les 15 s<br>"
            "Avant la passe : %{customdata[6]} % · <b>valeur ajoutée : %{customdata[7]} pts</b><br>"
            "Arrivée à %{customdata[8]} m du but · %{customdata[9]} adversaire(s) éliminé(s)<br>"
            "%{customdata[10]} défenseur(s) encore entre le ballon et le but<br>"
            "Suite de l'action : %{customdata[11]}<extra></extra>"
        ),
    ))
    return fig


def figure_freeze_frame(passe: pd.Series, joueurs: pd.DataFrame, vmax: float) -> go.Figure:
    """Positions des joueurs visibles au moment de la passe (StatsBomb 360)."""
    fig = terrain(hauteur=520)

    zone = passe["zone_visible"]
    if zone is not None and len(zone) >= 6:
        xs, ys = list(zone[0::2]), list(zone[1::2])
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", mode="lines", hoverinfo="skip",
                                 name="Champ de la caméra",
                                 fillcolor="rgba(61,209,255,0.07)",
                                 line=dict(color="rgba(61,209,255,0.35)", width=1, dash="dot")))

    couleur = couleur_danger([passe["danger"]], vmax)[0]
    fig.add_annotation(x=passe["x_fin"], y=passe["y_fin"], ax=passe["x_debut"], ay=passe["y_debut"],
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                       arrowhead=2, arrowsize=1.3, arrowwidth=3, arrowcolor=couleur)

    def points(masque, nom, couleur_pt, symbole, taille):
        sous = joueurs[masque]
        if sous.empty:
            return
        fig.add_trace(go.Scatter(x=sous["x"], y=sous["y"], mode="markers", name=nom,
                                 hovertemplate=f"{nom}<extra></extra>",
                                 marker=dict(size=taille, color=couleur_pt, symbol=symbole,
                                             line=dict(color="#0b0f13", width=1.5))))

    points(joueurs["coequipier"] & ~joueurs["passeur"] & ~joueurs["gardien"], passe["equipe"],
           COULEURS["attaque"], "circle", 13)
    points(joueurs["coequipier"] & joueurs["gardien"], f"Gardien ({passe['equipe']})",
           COULEURS["attaque"], "diamond", 13)
    points(~joueurs["coequipier"] & ~joueurs["gardien"], passe["adversaire"], COULEURS["defense"], "circle", 13)
    points(~joueurs["coequipier"] & joueurs["gardien"], f"Gardien ({passe['adversaire']})",
           COULEURS["defense"], "diamond", 13)
    points(joueurs["passeur"], f"Passeur : {passe['joueur']}", "#ffffff", "circle", 16)

    fig.add_trace(go.Scatter(x=[passe["x_fin"]], y=[passe["y_fin"]], mode="markers",
                             hovertemplate=f"Arrivée : {passe['receveur']}<extra></extra>", showlegend=False,
                             marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=couleur, width=2.5))))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=12),
                                                   bgcolor="rgba(0,0,0,0)"))
    return fig


def carte_chaleur(x, y, poids=None, titre: str | None = None, nx: int = 14, ny: int = 9,
                  hauteur: int = 460, couleur: str = "61,209,255", moyenne: bool = False,
                  valeurs=None, suffixe: str = " %") -> go.Figure:
    """
    Carte de chaleur sur le terrain.
    - par défaut : somme des poids par zone ;
    - moyenne=True : moyenne de `valeurs` par zone (zones avec peu de passes masquées).
    """
    fig = terrain(hauteur=hauteur, titre=titre)
    bx, by = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
    if moyenne:
        somme, _, _ = np.histogram2d(x, y, bins=[bx, by], weights=valeurs)
        compte, _, _ = np.histogram2d(x, y, bins=[bx, by])
        z = np.where(compte >= 30, somme / np.maximum(compte, 1), np.nan) * 100
    else:
        z, _, _ = np.histogram2d(x, y, bins=[bx, by], weights=poids)
    z = z.T
    fig.add_trace(go.Heatmap(
        x=(bx[:-1] + bx[1:]) / 2, y=(by[:-1] + by[1:]) / 2, z=z,
        colorscale=[[0, f"rgba({couleur},0.0)"], [0.15, f"rgba({couleur},0.18)"],
                    [0.6, f"rgba({couleur},0.55)"], [1, "rgba(255,209,102,0.9)"]],
        showscale=moyenne, xgap=1, ygap=1,
        colorbar=dict(ticksuffix=suffixe, thickness=10, len=0.75, outlinewidth=0) if moyenne else None,
        hovertemplate=("%{z:.1f}" + suffixe + "<extra></extra>") if moyenne else "%{z:.2f}<extra></extra>",
    ))
    # le terrain reste au-dessus de la carte
    fig.data = fig.data[::-1]
    for forme in fig.layout.shapes:
        forme.layer = "above"
    fig.layout.shapes[0].fillcolor = "rgba(0,0,0,0)"
    fig.update_layout(plot_bgcolor=COULEURS["pelouse"])
    return fig
