"""
Langue de l'application : français par défaut, anglais avec ?lang=en dans l'adresse.

La langue est relue dans l'adresse à chaque appel : elle est propre à chaque visiteur, même quand
plusieurs personnes utilisent l'application en même temps dans deux langues différentes.
Les données restent en français ; les libellés affichés sont traduits ici.
"""
from __future__ import annotations

import streamlit as st


def langue() -> str:
    return "en" if st.query_params.get("lang") == "en" else "fr"


def tr(fr: str, en: str) -> str:
    """Texte dans la langue du visiteur."""
    return en if langue() == "en" else fr


def selecteur_langue(conteneur=st) -> None:
    """Boutons FR | EN. Le choix est écrit dans l'adresse : un lien partagé garde la langue."""
    actuelle = langue()
    choix = conteneur.radio("Langue / Language", ["FR", "EN"], index=0 if actuelle == "fr" else 1,
                            horizontal=True, label_visibility="collapsed")
    if choix.lower() != actuelle:
        st.query_params["lang"] = choix.lower()
        st.rerun()


# --- Nombres : virgule et espaces en français, point et virgules en anglais ---------------------
def pct(x: float, decimales: int = 1) -> str:
    texte = f"{x * 100:.{decimales}f}"
    return f"{texte}%" if langue() == "en" else f"{texte.replace('.', ',')} %"


def nombre(x: float) -> str:
    texte = f"{x:,.0f}"
    return texte if langue() == "en" else texte.replace(",", " ")


def dec(x: float, decimales: int = 2) -> str:
    texte = f"{x:.{decimales}f}"
    return texte if langue() == "en" else texte.replace(".", ",")


def points(x: float) -> str:
    """Écart de probabilité en points, avec son signe : +2,3 pts / +2.3 pts."""
    return f"{dec(x * 100, 1) if x < 0 else '+' + dec(x * 100, 1)} pts"


# --- Libellés des données ---------------------------------------------------------------------
EQUIPES_EN = {
    "Allemagne": "Germany", "Angleterre": "England", "Arabie saoudite": "Saudi Arabia", "Argentine": "Argentina",
    "Australie": "Australia", "Belgique": "Belgium", "Brésil": "Brazil", "Cameroun": "Cameroon",
    "Corée du Sud": "South Korea", "Croatie": "Croatia", "Danemark": "Denmark", "Espagne": "Spain",
    "Japon": "Japan", "Maroc": "Morocco", "Mexique": "Mexico", "Pays de Galles": "Wales", "Pays-Bas": "Netherlands",
    "Pologne": "Poland", "Serbie": "Serbia", "Suisse": "Switzerland", "Sénégal": "Senegal", "Tunisie": "Tunisia",
    "Équateur": "Ecuador", "États-Unis": "United States",
}
PHASES_TOURNOI_EN = {
    "Phase de groupes": "Group stage", "Huitièmes de finale": "Round of 16", "Quarts de finale": "Quarter-finals",
    "Demi-finales": "Semi-finals", "Match pour la 3e place": "Third-place play-off", "Finale": "Final",
}
PHASES_JEU = {
    "fr": {"Regular Play": "Jeu placé", "From Counter": "Contre-attaque", "From Throw In": "Après une touche",
           "From Free Kick": "Après un coup franc", "From Corner": "Après un corner",
           "From Goal Kick": "Après un dégagement", "From Kick Off": "Après l'engagement",
           "From Keeper": "Relance du gardien", "Other": "Autre"},
    "en": {"Regular Play": "Open play", "From Counter": "Counter-attack", "From Throw In": "From a throw-in",
           "From Free Kick": "From a free kick", "From Corner": "From a corner",
           "From Goal Kick": "From a goal kick", "From Kick Off": "From kick-off",
           "From Keeper": "From the goalkeeper", "Other": "Other"},
}
HAUTEURS = {
    "fr": {"Ground Pass": "Passe à ras de terre", "Low Pass": "Passe à mi-hauteur", "High Pass": "Passe aérienne"},
    "en": {"Ground Pass": "Ground pass", "Low Pass": "Low pass", "High Pass": "High pass"},
}
MODELES_EN = {
    "Ancienne formule (poids à la main)": "Previous formula (hand-picked weights)",
    "Distance au but seule": "Distance to goal only", "Position seule (sans 360)": "Position only (no 360)",
    "Régression logistique": "Logistic regression", "Gradient boosting": "Gradient boosting",
}
VARIABLES_EN = {
    "Distance au but (arrivée)": "Distance to goal (end point)", "Angle de tir (arrivée)": "Shooting angle (end point)",
    "Défenseurs restants": "Defenders remaining", "Soutien offensif": "Attacking support",
    "Visibilité 360": "360 visibility", "Contexte de jeu": "Game context", "Technique de passe": "Pass technique",
    "Pression sur le receveur": "Pressure on the receiver", "Point de départ": "Starting point",
    "Adversaires éliminés": "Opponents bypassed", "Passe dans le dos de la défense": "Pass in behind the defence",
    "Pression sur le passeur": "Pressure on the passer", "Coup de pied arrêté": "Set piece",
    "Progression vers le but": "Progression towards goal", "Cône de tir obstrué": "Blocked shooting cone",
}
ZONES_EN = {"Propre moitié": "Own half", "Milieu offensif": "Attacking midfield", "Dernier tiers": "Final third",
            "Surface et abords": "Box and surroundings"}


def libelle(texte: str, traductions: dict) -> str:
    """Traduction anglaise d'une valeur des données (inchangée en français ou si elle n'a pas de traduction)."""
    return traductions.get(texte, texte) if langue() == "en" else texte
