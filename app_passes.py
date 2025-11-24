import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.patches as mpatches

# ------------------------------------
# ⚽ Fonction pour dessiner le terrain 
# ------------------------------------
def create_pitch(field_size=(106,68)):
    lenx = field_size[0]
    leny = field_size[1]/2
    
    fig=plt.figure(figsize=(12,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_facecolor('mediumseagreen')

    #Pitch Outline & Centre Line
    plt.plot([0,0],[-leny,leny], color="black")
    plt.plot([0,lenx],[leny,leny], color="black")
    plt.plot([lenx,lenx],[leny,-leny], color="black")
    plt.plot([lenx,0],[-leny,-leny], color="black")
    plt.plot([lenx/2,lenx/2],[leny,-leny], color="black")
    
    #Left Penalty Area
    plt.plot([16.5,16.5],[-20.15,20.15],color="black")
    plt.plot([0,16.5],[20.15,20.15],color="black")
    plt.plot([16.5,0],[-20.15,-20.15],color="black")
    
    #Right Penalty Area
    plt.plot([lenx-16.5,lenx],[20.15,20.15],color="black")
    plt.plot([lenx-16.5,lenx-16.5],[-20.15,20.15],color="black")
    plt.plot([lenx-16.5,106],[-20.15,-20.15],color="black")
    
    #Left 6-yard Box
    plt.plot([0,5.5],[9,9],color="black")
    plt.plot([5.5,5.5],[9,-9],color="black")
    plt.plot([5.5,0.5],[-9,-9],color="black")
    
    #Right 6-yard Box
    plt.plot([106,lenx-5.5],[9,9],color="black")
    plt.plot([lenx-5.5,lenx-5.5],[9,-9],color="black")
    plt.plot([lenx-5.5,106],[-9,-9],color="black")
    
    #Circles
    centreCircle = plt.Circle((lenx/2,0),9.15,color="black",fill=False)
    centreSpot = plt.Circle((lenx/2,0),0.8,color="black")
    leftPenSpot = plt.Circle((11,0),0.8,color="black")
    rightPenSpot = plt.Circle((lenx-11,0),0.8,color="black")
    
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    #Arcs
    leftArc = Arc((11,0),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="black")
    rightArc = Arc((lenx-11,0),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="black")

    ax.add_patch(leftArc)
    ax.add_patch(rightArc)   
    
    return fig, ax


# ------------------------
# Configuration de la page
# ------------------------
st.set_page_config(page_title="Visualisation des passes dangereuses", layout="wide")
st.title("⚽️ Création d'un indicateur de dangerosité des passes et affichage sur le terrain ⚽️")

st.markdown("""
Cette application permet d'explorer visuellement les **passes dangereuses** réalisées au cours d’un match de football. 
Elle peut être utilisée pour analyser les failles défensives ou comprendre les stratégies offensives d’une équipe. 
Un exemple simple est disponible en fin de page. Elle s’appuie sur les données disponibles au lien suivant : https://github.com/metrica-sports/sample-data.

Un indicateur de dangerosité a été préalablement calculé pour chaque passe, basé notamment sur :
- la progression vers le but adverse,
- la position des joueurs adverses,
- l’angle de la passe par rapport au but,
- la distance restante jusqu’au but,
- la pression des défenseurs sur le receveur,
- le nombre de joueurs adverses éliminés par la passe.

Il est possible de filtrer les passes selon plusieurs critères :
- dangerosité minimale,
- top X% des passes les plus dangereuses,
- mi-temps,
- équipe spécifique.

Les passes correspondantes sont affichées sur un terrain de football, sous forme de flèches indiquant la direction, la distance et la zone d’impact de chaque action.
""")

# ------------------------
# 📂 Chargement des données
# ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("pass_features_danger_scores.csv", sep=',')

df = load_data()


# ------------------------
# Contrôles utilisateur
# ------------------------

# Dangerosité minimum
min_danger = st.slider(
    "Dangerosité minimum des passes à afficher",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.01
)

# Nombre de passes
top_percent = st.slider(
    "Top % des passes les plus dangereuses à afficher",
    min_value=1,
    max_value=100,
    value=30,
    step=1
)

# Sélection de la mi-temps
period = st.selectbox(
    "Choisir la mi-temps",
    options=[1, 2],
    index=0
)

teams = ["Toutes"] + sorted(df["Team"].unique())

selected_team = st.selectbox(
    "Filtrer sur une équipe (optionnel)",
    options=teams,
    index=0
)

# Filtrage dangerosité + mi-temps
filtered = df[df["danger_score"] >= min_danger]
filtered = filtered[filtered["Period"] == period]

# Filtre d'équipe si choisi
if selected_team != "Toutes":
    filtered = filtered[filtered["Team"] == selected_team]

# Top % 
n_selected = max(1, int(len(filtered) * (top_percent / 100)))
filtered = filtered.sort_values("danger_score", ascending=False).head(n_selected)


# ------------------------
# Affichage sur terrain
# ------------------------
fig, ax = create_pitch()

# Normalisation des coordonnées
def scale_x(x): return x * 106
def scale_y(y): return (y - 0.5) * 68  

team_colors = {
    "Home": "blue",
    "Away": "red"
}

legend_handles = [
    mpatches.Patch(color=team_colors["Home"], label="Home"),
    mpatches.Patch(color=team_colors["Away"], label="Away")
]


for _, row in filtered.iterrows():
    x1 = scale_x(row["Start X"])
    y1 = scale_y(row["Start Y"])
    x2 = scale_x(row["End X"])
    y2 = scale_y(row["End Y"])

    # Couleur selon équipe
    team = row["Team"]
    color = team_colors.get(team, "yellow")

    ax.arrow(
        x1, y1,
        x2 - x1, y2 - y1,
        length_includes_head=True,
        head_width=1.2,
        head_length=2,
        color=color,
        alpha=0.9
    )

ax.set_xlim(0, 106)
ax.set_ylim(-34, 34)
ax.axis("off")

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    frameon=False
)

st.pyplot(fig)

st.markdown("""
**Exemple d'utilisation :**
            
On s’intéresse aux passes ayant un score de dangerosité supérieur à 0.65 durant la première mi-temps, et on affiche le top 30% de ces passes.
On constate grâce à cet outil que l’équipe "Home" a réussi plus de passes dangereuses que l’équipe "Away", et celles-ci viennent majoritairement de l’aile droite. Cependant, l’équipe "Away" a quand même réussi des passes dangereuses sur le côté gauche, ce qui pourrait indiquer une faiblesse défensive à cet endroit précis. 
Au final, on constate que la majorité des passes dangereuses se concentrent sur la même zone du terrain, témoignant d’un côté plutôt ouvert par les défenses adverses et d’un autre côté plus verrouillé.      
""")