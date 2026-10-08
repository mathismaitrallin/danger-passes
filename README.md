# Dangerosité des passes · Coupe du monde 2022

**Quelles passes mettent vraiment l'adversaire en danger ?** Pour chacune des 47 329 passes réussies de la Coupe du monde 2022, un modèle estime la probabilité que l'équipe tire dans les 15 secondes qui suivent. Il s'appuie sur l'endroit où arrive la passe et sur la position des joueurs au moment de la passe (données StatsBomb 360).

👉 **Application interactive :** https://danger-passes-mathis-maitrallin.streamlit.app

- Deux façons de classer les passes : **danger** (probabilité de tir) ou **valeur ajoutée** (gain par rapport à la situation avant la passe)
- Carte des passes, filtrable par match, équipe, joueur et période
- « Revoir une passe » : position des joueurs au moment de la passe et explication du score
- Classement des passeurs, danger créé et concédé par équipe
- Page « Méthode & validation » avec tous les résultats ci-dessous

## Résultats

Validation croisée **groupée par match** (5 blocs) : chaque passe est notée par un modèle qui n'a jamais vu son match. L'AUC mesure la capacité à bien classer les passes (0,5 = hasard, 1 = parfait).

| Approche | AUC (toutes les passes) | AUC (dernier tiers) |
|---|---|---|
| Distance au but seule | 0,829 | 0,742 |
| Position seule (sans 360) | 0,827 | 0,748 |
| Gradient boosting | 0,834 | 0,752 |
| **Régression logistique (retenue)** | **0,836** | **0,753** |

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

## Limites

- **Joueurs hors champ** : les données 360 viennent des images TV, les joueurs hors cadre ne sont pas connus.
- **Passes ratées non notées** : un modèle de risque compléterait l'analyse (rapport risque / danger).
