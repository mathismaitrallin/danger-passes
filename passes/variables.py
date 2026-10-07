"""
Construction des variables de chaque passe à partir des événements et des freeze frames 360.

Conventions :
- toutes les coordonnées sont converties en mètres sur un terrain de 105 x 68 ;
- l'équipe qui fait la passe attaque toujours vers la droite (x = 105), comme chez StatsBomb ;
- y = 0 correspond au haut du terrain (côté gauche de l'équipe qui attaque).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .donnees import PHASES_FR, equipe_fr

LONGUEUR, LARGEUR = 105.0, 68.0
ECHELLE_X, ECHELLE_Y = LONGUEUR / 120, LARGEUR / 80
BUT = np.array([LONGUEUR, LARGEUR / 2])
DEMI_BUT = 7.32 / 2
FENETRE_TIR = 15.0      # secondes : une passe est "suivie d'un tir" si le tir arrive dans ce délai
DISTANCE_MAX = 20.0     # plafond des distances au plus proche adversaire (joueurs hors champ inconnus)

CARTONS_ROUGES = {"Red Card", "Second Yellow"}


def _metres(loc) -> np.ndarray:
    return np.array([loc[0] * ECHELLE_X, loc[1] * ECHELLE_Y])


def _secondes(timestamp: str) -> float:
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _angle_but(point: np.ndarray) -> float:
    """Angle (en degrés) sous lequel on voit le but depuis un point : grand = position de tir ouverte."""
    dx = LONGUEUR - point[0]
    a1 = np.arctan2(BUT[1] - DEMI_BUT - point[1], dx)
    a2 = np.arctan2(BUT[1] + DEMI_BUT - point[1], dx)
    return float(np.degrees(abs(a2 - a1)))


def _dans_cone_tir(points: np.ndarray, sommet: np.ndarray) -> np.ndarray:
    """Points situés dans le triangle formé par `sommet` et les deux poteaux (le 'cône de tir')."""
    a, b, c = sommet, np.array([LONGUEUR, BUT[1] - DEMI_BUT]), np.array([LONGUEUR, BUT[1] + DEMI_BUT])
    signe = lambda p, q, r: (p[:, 0] - r[0]) * (q[1] - r[1]) - (q[0] - r[0]) * (p[:, 1] - r[1])
    d1, d2, d3 = signe(points, a, b), signe(points, b, c), signe(points, c, a)
    negatif = (d1 < 0) | (d2 < 0) | (d3 < 0)
    positif = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(negatif & positif)


def _dans_surface(points: np.ndarray) -> np.ndarray:
    return (points[:, 0] >= LONGUEUR - 16.5) & (np.abs(points[:, 1] - LARGEUR / 2) <= 20.16)


def _decalages_periodes(evenements: list) -> dict[int, float]:
    """Temps écoulé avant chaque période (en secondes), pour avoir un temps de jeu continu."""
    fins = {}
    for e in evenements:
        if e["type"]["name"] == "Half End" and e["period"] <= 4:
            fins[e["period"]] = max(fins.get(e["period"], 0.0), _secondes(e["timestamp"]))
    decalages, cumul = {}, 0.0
    for p in sorted(fins):
        decalages[p] = cumul
        cumul += fins[p]
    decalages["fin"] = cumul
    return decalages


def _noms_joueurs(compositions: list) -> dict[int, str]:
    noms = {}
    for equipe in compositions:
        for j in equipe["lineup"]:
            noms[j["player_id"]] = j.get("player_nickname") or j["player_name"]
    return noms


def minutes_jouees(evenements: list, match_id: int, noms: dict[int, str]) -> pd.DataFrame:
    """Minutes jouées par joueur (titulaires, remplacements et expulsions)."""
    decalages = _decalages_periodes(evenements)
    t_abs = lambda e: decalages.get(e["period"], 0.0) + _secondes(e["timestamp"])

    entree, sortie, equipe = {}, {}, {}
    for e in evenements:
        nom_type = e["type"]["name"]
        if e["period"] > 4:
            continue
        if nom_type == "Starting XI":
            for j in e["tactics"]["lineup"]:
                entree[j["player"]["id"]] = 0.0
                equipe[j["player"]["id"]] = e["team"]["name"]
        elif nom_type == "Substitution":
            sortie[e["player"]["id"]] = t_abs(e)
            remplacant = e["substitution"]["replacement"]["id"]
            entree[remplacant] = t_abs(e)
            equipe[remplacant] = e["team"]["name"]
        else:
            carton = (e.get("foul_committed") or e.get("bad_behaviour") or {}).get("card", {}).get("name")
            if carton in CARTONS_ROUGES and "player" in e:
                sortie[e["player"]["id"]] = t_abs(e)

    lignes = [
        {
            "match_id": match_id,
            "joueur_id": pid,
            "joueur": noms.get(pid, str(pid)),
            "equipe": equipe_fr(equipe[pid]),
            "minutes": (sortie.get(pid, decalages["fin"]) - debut) / 60,
        }
        for pid, debut in entree.items()
    ]
    return pd.DataFrame(lignes)


def _tirs(evenements: list, decalages: dict) -> list[dict]:
    return [
        {
            "equipe": e["team"]["name"],
            "possession": e["possession"],
            "periode": e["period"],
            "t": _secondes(e["timestamp"]),
            "xg": e["shot"].get("statsbomb_xg", 0.0),
            "but": e["shot"]["outcome"]["name"] == "Goal",
        }
        for e in evenements
        if e["type"]["name"] == "Shot" and e["period"] <= 4
    ]


def _type_passe(p: dict) -> str:
    type_sb = p.get("type", {}).get("name")
    return {
        "Corner": "Corner", "Free Kick": "Coup franc", "Throw-in": "Touche",
        "Goal Kick": "Dégagement", "Kick Off": "Engagement",
    }.get(type_sb, "Jeu courant")


def extraire_match(evenements: list, freeze_frames: list, compositions: list, match: dict):
    """
    Renvoie trois tables pour un match :
    - passes : une ligne par passe réussie disposant d'un freeze frame 360 ;
    - joueurs_ff : positions des joueurs visibles au moment de chaque passe ;
    - minutes : minutes jouées par joueur.
    """
    match_id = match["match_id"]
    noms = _noms_joueurs(compositions)
    decalages = _decalages_periodes(evenements)
    tirs = _tirs(evenements, decalages)
    ff_par_id = {f["event_uuid"]: f for f in freeze_frames}

    domicile = match["home_team"]["home_team_name"]
    exterieur = match["away_team"]["away_team_name"]

    passes, positions = [], []
    for e in evenements:
        if e["type"]["name"] != "Pass" or e["period"] > 4:
            continue
        p = e["pass"]
        if "outcome" in p:                       # on ne garde que les passes réussies
            continue
        ff = ff_par_id.get(e["id"])
        if ff is None or not ff.get("freeze_frame"):
            continue

        debut, fin = _metres(e["location"]), _metres(p["end_location"])
        t = _secondes(e["timestamp"])
        equipe = e["team"]["name"]

        # --- Positions des joueurs visibles au moment de la passe ---
        joueurs = [(_metres(j["location"]), j["teammate"], j["keeper"], j["actor"]) for j in ff["freeze_frame"]]
        adv = np.array([pos for pos, coeq, _, _ in joueurs if not coeq]).reshape(-1, 2)
        adv_champ = np.array([pos for pos, coeq, gk, _ in joueurs if not coeq and not gk]).reshape(-1, 2)
        coeq = np.array([pos for pos, c, _, acteur in joueurs if c and not acteur]).reshape(-1, 2)

        dist_but_debut = float(np.linalg.norm(BUT - debut))
        dist_but_fin = float(np.linalg.norm(BUT - fin))

        if len(adv):
            d_fin = np.linalg.norm(adv - fin, axis=1)
            d_debut = np.linalg.norm(adv - debut, axis=1)
            dist_adv_fin = float(min(d_fin.min(), DISTANCE_MAX))
            dist_adv_debut = float(min(d_debut.min(), DISTANCE_MAX))
            adv_5m_fin = int((d_fin < 5).sum())
            adv_5m_debut = int((d_debut < 5).sum())
        else:
            dist_adv_fin = dist_adv_debut = DISTANCE_MAX
            adv_5m_fin = adv_5m_debut = 0

        if len(adv_champ):
            x_adv = adv_champ[:, 0]
            elimines = int(((x_adv > debut[0]) & (x_adv < fin[0])).sum()) if fin[0] > debut[0] else 0
            dist_but_adv = np.linalg.norm(BUT - adv_champ, axis=1)
            restants = int((dist_but_adv < dist_but_fin).sum())
            restants_debut = int((dist_but_adv < dist_but_debut).sum())
            # > 0 : la passe arrive derrière le dernier défenseur visible (appel en profondeur)
            derriere_defense = float(np.clip(fin[0] - x_adv.max(), -DISTANCE_MAX, DISTANCE_MAX))
            cone = int(_dans_cone_tir(adv_champ, fin).sum())
            adv_surface = int(_dans_surface(adv_champ).sum())
        else:
            elimines, restants, restants_debut, cone, adv_surface = 0, 0, 0, 0, 0
            derriere_defense = -DISTANCE_MAX
        gardien_adv = np.array([pos for pos, c, gk, _ in joueurs if not c and gk]).reshape(-1, 2)
        gardien_dans_cone = bool(len(gardien_adv) and _dans_cone_tir(gardien_adv, fin).any())

        vecteur_passe, vecteur_but = fin - debut, BUT - debut
        direction = float(np.dot(vecteur_passe, vecteur_but) /
                          (np.linalg.norm(vecteur_passe) * np.linalg.norm(vecteur_but) + 1e-9))

        # --- Cible : un tir de la même équipe, dans la même possession, dans les 15 s ---
        suivants = [s for s in tirs if s["equipe"] == equipe and s["possession"] == e["possession"]
                    and s["periode"] == e["period"] and 0 < s["t"] - t <= FENETRE_TIR]

        longueur = float(np.linalg.norm(vecteur_passe))
        passes.append({
            "passe_id": e["id"],
            "match_id": match_id,
            "periode": e["period"],
            "minute": e["minute"],
            "seconde": e["second"],
            "t_abs": decalages.get(e["period"], 0.0) + t,
            "equipe": equipe_fr(equipe),
            "adversaire": equipe_fr(exterieur if equipe == domicile else domicile),
            "joueur_id": e["player"]["id"],
            "joueur": noms.get(e["player"]["id"], e["player"]["name"]),
            "receveur": noms.get(p["recipient"]["id"], p["recipient"]["name"]) if "recipient" in p else "",
            "poste": e.get("position", {}).get("name", ""),
            "x_debut": debut[0], "y_debut": debut[1], "x_fin": fin[0], "y_fin": fin[1],
            "type_passe": _type_passe(p),
            "phase_jeu": e["play_pattern"]["name"],
            "hauteur": p.get("height", {}).get("name", "Ground Pass"),
            "en_profondeur": p.get("technique", {}).get("name") == "Through Ball",
            "centre": bool(p.get("cross")),
            "renversement": bool(p.get("switch")),
            "passe_en_retrait": bool(p.get("cut_back")),
            "passeur_sous_pression": bool(e.get("under_pressure")),
            "duree": e.get("duration") or np.nan,
            # variables géométriques
            "longueur": longueur,
            "direction_but": direction,
            "dist_but_debut": dist_but_debut,
            "dist_but_fin": dist_but_fin,
            "progression": dist_but_debut - dist_but_fin,
            "angle_but_fin": _angle_but(fin),
            "ecart_axe_fin": abs(fin[1] - LARGEUR / 2),
            "dans_surface_fin": bool(fin[0] >= LONGUEUR - 16.5 and abs(fin[1] - LARGEUR / 2) <= 20.16),
            # variables issues du 360
            "adv_visibles": int(len(adv)),
            "adv_elimines": elimines,
            "defenseurs_restants": restants,
            "dist_adv_fin": dist_adv_fin,
            "adv_5m_fin": adv_5m_fin,
            "dist_adv_debut": dist_adv_debut,
            # situation du passeur avant la passe (sert au modèle de "valeur avant la passe")
            "angle_but_debut": _angle_but(debut),
            "ecart_axe_debut": abs(debut[1] - LARGEUR / 2),
            "dans_surface_debut": bool(debut[0] >= LONGUEUR - 16.5 and abs(debut[1] - LARGEUR / 2) <= 20.16),
            "defenseurs_restants_debut": restants_debut,
            "adv_5m_debut": adv_5m_debut,
            "coequipiers_devant_debut": int((coeq[:, 0] > debut[0]).sum()) if len(coeq) else 0,
            "coequipiers_devant": int((coeq[:, 0] > fin[0]).sum()) if len(coeq) else 0,
            "derriere_defense": derriere_defense,
            "adv_cone_tir": cone,
            "gardien_dans_cone": gardien_dans_cone,
            "adv_surface": adv_surface,
            "coequipiers_surface": int(_dans_surface(coeq).sum()) if len(coeq) else 0,
            "zone_visible": [round(v * (ECHELLE_X if i % 2 == 0 else ECHELLE_Y), 2)
                             for i, v in enumerate(ff["visible_area"])],
            # cible et informations sur la suite de l'action
            "tir_15s": bool(suivants),
            "xg_15s": max((s["xg"] for s in suivants), default=0.0),
            "but_15s": any(s["but"] for s in suivants),
        })
        for pos, est_coeq, gardien, acteur in joueurs:
            positions.append({
                "passe_id": e["id"], "x": round(float(pos[0]), 2), "y": round(float(pos[1]), 2),
                "coequipier": est_coeq, "gardien": gardien, "passeur": acteur,
            })

    nb_passes_reussies = sum(1 for e in evenements if e["type"]["name"] == "Pass"
                             and e["period"] <= 4 and "outcome" not in e["pass"])
    return pd.DataFrame(passes), pd.DataFrame(positions), minutes_jouees(evenements, match_id, noms), nb_passes_reussies


def infos_match(match: dict) -> dict:
    dom = equipe_fr(match["home_team"]["home_team_name"])
    ext = equipe_fr(match["away_team"]["away_team_name"])
    phase = PHASES_FR.get(match["competition_stage"]["name"], match["competition_stage"]["name"])
    return {
        "match_id": match["match_id"],
        "date": match["match_date"],
        "phase": phase,
        "domicile": dom,
        "exterieur": ext,
        "score": f'{match["home_score"]}-{match["away_score"]}',
        "libelle": f'{dom} {match["home_score"]}-{match["away_score"]} {ext}',
    }
