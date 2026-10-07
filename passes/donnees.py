"""
Téléchargement et lecture des données StatsBomb Open Data.

Source : https://github.com/statsbomb/open-data (données gratuites, attribution requise).
Les fichiers bruts sont mis en cache localement pour ne les télécharger qu'une fois.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

URL_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Coupe du monde 2022 : identifiants StatsBomb
COMPETITION_ID = 43
SAISON_ID = 106


def _telecharger_json(chemin_distant: str, dossier_cache: Path) -> dict | list:
    """Télécharge un fichier JSON (avec cache disque et quelques tentatives)."""
    fichier = dossier_cache / chemin_distant
    if fichier.exists():
        return json.loads(fichier.read_text(encoding="utf-8"))

    fichier.parent.mkdir(parents=True, exist_ok=True)
    for essai in range(4):
        try:
            r = requests.get(f"{URL_BASE}/{chemin_distant}", timeout=60)
            r.raise_for_status()
            fichier.write_bytes(r.content)
            return r.json()
        except requests.RequestException:
            if essai == 3:
                raise
            time.sleep(2 * (essai + 1))


def charger_matchs(dossier_cache: Path) -> list[dict]:
    return _telecharger_json(f"matches/{COMPETITION_ID}/{SAISON_ID}.json", dossier_cache)


def charger_match(match_id: int, dossier_cache: Path) -> tuple[list, list, list]:
    """Renvoie (événements, freeze frames 360, compositions) d'un match."""
    evenements = _telecharger_json(f"events/{match_id}.json", dossier_cache)
    freeze_frames = _telecharger_json(f"three-sixty/{match_id}.json", dossier_cache)
    compositions = _telecharger_json(f"lineups/{match_id}.json", dossier_cache)
    return evenements, freeze_frames, compositions


# Noms d'équipes affichés en français
EQUIPES_FR = {
    "Argentina": "Argentine", "Australia": "Australie", "Belgium": "Belgique",
    "Brazil": "Brésil", "Cameroon": "Cameroun", "Canada": "Canada",
    "Costa Rica": "Costa Rica", "Croatia": "Croatie", "Denmark": "Danemark",
    "Ecuador": "Équateur", "England": "Angleterre", "France": "France",
    "Germany": "Allemagne", "Ghana": "Ghana", "Iran": "Iran", "Japan": "Japon",
    "Mexico": "Mexique", "Morocco": "Maroc", "Netherlands": "Pays-Bas",
    "Poland": "Pologne", "Portugal": "Portugal", "Qatar": "Qatar",
    "Saudi Arabia": "Arabie saoudite", "Senegal": "Sénégal", "Serbia": "Serbie",
    "South Korea": "Corée du Sud", "Spain": "Espagne", "Switzerland": "Suisse",
    "Tunisia": "Tunisie", "United States": "États-Unis", "Uruguay": "Uruguay",
    "Wales": "Pays de Galles",
}

PHASES_FR = {
    "Group Stage": "Phase de groupes", "Round of 16": "Huitièmes de finale",
    "Quarter-finals": "Quarts de finale", "Semi-finals": "Demi-finales",
    "3rd Place Final": "Match pour la 3e place", "Final": "Finale",
}


def equipe_fr(nom: str) -> str:
    return EQUIPES_FR.get(nom, nom)
