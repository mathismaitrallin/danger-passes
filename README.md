# Dangerosité des passes · Coupe du monde 2022

**Quelles passes mettent vraiment l'adversaire en danger ?** Pour chacune des 47 329 passes réussies de la Coupe du monde 2022, un modèle estime la probabilité que l'équipe tire dans les 15 secondes qui suivent. Il s'appuie sur l'endroit où arrive la passe et sur la position des joueurs au moment de la passe (données StatsBomb 360).

👉 **Application interactive :** https://danger-passes-mathis-maitrallin.streamlit.app

- Deux façons de classer les passes : **danger** (probabilité de tir) ou **valeur ajoutée** (gain par rapport à la situation avant la passe)
- Carte des passes, filtrable par match, équipe, joueur et période
- « Revoir une passe » : position des joueurs au moment de la passe et explication du score
- Classement des passeurs, danger créé et concédé par équipe
- Page « Méthode & validation » avec tous les résultats ci-dessous

## Résultats

Validation croisée **groupée par match** (5 plis) : chaque passe est notée par un modèle qui n'a jamais vu son match. L'AUC mesure la capacité à bien classer les passes (0,5 = hasard, 1 = parfait).

| Approche | AUC (toutes les passes) | AUC (dernier tiers) |
|---|---|---|
| Ancienne formule (poids choisis à la main) | 0,718 | 0,583 |
| Distance au but seule | 0,829 | 0,742 |
| Position seule (sans 360) | 0,827 | 0,748 |
| Gradient boosting | 0,834 | 0,752 |
| **Régression logistique (retenue)** | **0,836** | **0,753** |

- **La première version du projet ne distinguait presque pas les passes dangereuses dans le dernier tiers** (0,58). Gain du nouveau modèle : +0,118 d'AUC, IC 95 % [0,108 ; 0,129] (bootstrap sur les matchs).
- **L'endroit où arrive la passe explique l'essentiel du danger.**
- **Le contexte 360 apporte un gain modeste mais significatif** : +0,009 d'AUC, IC 95 % [0,005 ; 0,013].
- **Probabilités bien calibrées** : pour les 10 % de passes les plus dangereuses, le modèle annonce 33 % et un tir suit dans 33 % des cas.
- La régression logistique est retenue : aussi précise que le gradient boosting, meilleure log loss, plus simple à expliquer.

## Valeur ajoutée : récompenser aussi la progression

Le danger mesure la proximité d'un tir : une passe qui casse une ligne au milieu de terrain reste peu dangereuse. Un second modèle estime donc la probabilité de tir **avant la passe**, à partir de la seule situation du passeur (position, pression, défenseurs entre lui et le but, phase de jeu). Il atteint une AUC de 0,795 en validation par match.

> **valeur ajoutée = danger après la passe − probabilité de tir avant la passe**

Elle vaut zéro en moyenne et récompense les passes qui cassent des lignes :

| Zone d'arrivée | 0 adversaire éliminé | 3 ou plus |
|---|---|---|
| Milieu offensif | −2,3 pts | **+2,3 pts** |
| Dernier tiers | −3,4 pts | **+5,0 pts** |
| Surface et abords | +2,6 pts | **+10,1 pts** |

Limite : dans sa propre moitié, le gain reste quasi nul (la probabilité de tir dans les 15 s y est très faible avant comme après). Valoriser la relance demanderait une cible à plus long terme.

## Méthode

1. **Données** : [StatsBomb Open Data](https://github.com/statsbomb/open-data), 64 matchs, événements + freeze frames 360. 84 % des passes réussies ont une image 360 ; les autres (ralentis, plans serrés) sont exclues.
2. **Variables** (en mètres, au moment de la passe uniquement) :
   - position : distance et angle de tir au point d'arrivée, progression vers le but ;
   - contexte 360 : adversaires éliminés, défenseurs entre le ballon et le but, arrivée dans le dos du dernier défenseur, défenseurs dans le cône de tir, espace autour du receveur et du passeur, soutien offensif ;
   - passe : hauteur, passe en profondeur, centre, passe en retrait, coup de pied arrêté, contre-attaque.
3. **Cible** : tir de la même équipe, dans la même possession, dans les 15 secondes (6,7 % des passes).
4. **Modèles comparés** : ancienne formule, distance seule, modèle de position seule, régression logistique, gradient boosting.
5. **Interprétation** : importance par permutation, par famille de variables.

## Structure

```
├── app_passes.py            application Streamlit
├── construire_donnees.py    pipeline complet : téléchargement → variables → modèle → data/
├── analyse.ipynb            notebook de présentation des résultats
├── passes/
│   ├── donnees.py           téléchargement StatsBomb (avec cache)
│   ├── variables.py         variables de chaque passe (géométrie + 360)
│   ├── modele.py            modèles, validation par match, calibration, importance
│   └── terrain.py           dessin du terrain et des passes (Plotly)
├── data/                    données prêtes pour l'application (~9 Mo)
└── ancienne_version_metrica/  première version du projet (données Metrica)
```

## Reproduire

```bash
pip install -r requirements.txt
python construire_donnees.py   # télécharge ~80 Mo depuis GitHub, environ 5 minutes
streamlit run app_passes.py
```

## Limites

- **Joueurs hors champ** : les données 360 viennent des images TV, les joueurs hors cadre ne sont pas connus.
- **Une photo, pas un film** : positions au moment de la passe, pas de la réception.
- **Un seul tournoi** : prochaine étape, tester le modèle sur l'Euro 2024 ou la Ligue 1 (aussi disponibles en 360).
- **Passes ratées non notées** : un modèle de risque compléterait l'analyse (rapport risque / danger).

## Première version

La version initiale (dossier `ancienne_version_metrica/`) utilisait un match anonymisé Metrica et un score à poids fixés à la main. Elle a été refaite pour trois raisons : les poids n'étaient pas validés, un terme de pression avait le mauvais signe, et l'axe vertical du terrain était inversé dans l'affichage.

---

Données : StatsBomb Open Data (Coupe du monde 2022, 360). Projet réalisé par Mathis Maitrallin, Data Scientist.
