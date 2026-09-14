import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_pie_charts():
    # 1. Chargement du fichier descriptif (vérité terrain non normalisée)
    # On précise sep=";" car c'est le standard dans ce projet
    df_desc = pd.read_csv("data/Descriptif_patients(Sheet1).csv", sep=";")
    
    # Nettoyage de l'âge dans le fichier descriptif (peut contenir des virgules au lieu de points)
    df_desc['Age_at_disease'] = pd.to_numeric(
        df_desc['Age_at_disease'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    
    # Création des tranches d'âge standard
    df_desc['Age_Group'] = pd.cut(
        df_desc['Age_at_disease'], 
        bins=[0, 50, 70, 120], 
        labels=['<50', '50-70', '>70']
    ).astype(str).replace('nan', 'Inconnu')
    
    df_desc = df_desc[['id', 'Gender', 'classe_name', 'Age_Group']]

    # 2. Définition des bases de données à analyser
    bases_de_donnees = [
        {
            "nom": "Avec tous les patients",
            "fichier": "data/global_excel_resampled_scaled.csv"
        },
        {
            "nom": "Avec les patients ayant 3 phases mesurées",
            "fichier": "data/global_excel_resampled_normalized_flattened.csv"
        }
    ]

    # Configuration de la figure (2 lignes, 3 colonnes)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    fig.suptitle("Proportions avant et après la sélection des patients ayant 3 phases mesurées", fontsize=22, fontweight='bold')

    # Palettes de couleurs intenses (ton sur ton)
    colors_gender = sns.color_palette("Blues", n_colors=4)[2:]    # Bleus intenses
    colors_cancer = sns.color_palette("Oranges", n_colors=5)[2:]  # Oranges intenses (il y a 3 classes : CHC, CCK, Mixtes)
    colors_age = sns.color_palette("Greens", n_colors=5)[2:]      # Verts intenses

    for i, base in enumerate(bases_de_donnees):
        print(f"Traitement de {base['fichier']}...")
        
        # Chargement de la base
        df_base = pd.read_csv(base['fichier'], sep=";")
        
        # On ne garde que les IDs uniques pour ne pas compter un patient plusieurs fois 
        # (particulièrement utile pour la base non-aplatie qui a 3 lignes par patient)
        df_unique_ids = df_base[['id']].drop_duplicates()
        
        # Jointure avec le descriptif patient
        df_merged = df_unique_ids.merge(df_desc, on='id', how='left')
        
        # Remplissage des valeurs manquantes si la jointure échoue sur certains IDs (ex: patients synthétiques SMOTE)
        df_merged.fillna('Inconnu', inplace=True)
        
        nom_affichage = base['nom']

        # -- Graphe 1: Homme / Femme --
        counts_gender = df_merged['Gender'].value_counts()
        axes[i, 0].pie(counts_gender, labels=counts_gender.index, autopct='%1.1f%%', startangle=90, colors=colors_gender, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 15})
        axes[i, 0].set_title(f"Sexe\n{nom_affichage}", fontweight='bold', fontsize=14)

        # -- Graphe 2: Type de Cancer --
        counts_cancer = df_merged['classe_name'].value_counts()
        axes[i, 1].pie(counts_cancer, labels=counts_cancer.index, autopct='%1.1f%%', startangle=90, colors=colors_cancer, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 15})
        axes[i, 1].set_title(f"Type de Cancer\n{nom_affichage}", fontweight='bold', fontsize=14)

        # -- Graphe 3: Tranche d'Âge --
        counts_age = df_merged['Age_Group'].value_counts()
        axes[i, 2].pie(counts_age, labels=counts_age.index, autopct='%1.1f%%', startangle=90, colors=colors_age, wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 15})
        axes[i, 2].set_title(f"Tranche d'âge\n{nom_affichage}", fontweight='bold', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarde
    os.makedirs(os.path.join("plots", "visualisation"), exist_ok=True)
    filename = os.path.join("plots", "visualisation", "comparaison_bases_camemberts.png")
    plt.savefig(filename, dpi=300)
    print(f"\nGraphique sauvegardé sous : {filename}")
    
    # Affichage
    plt.show()

if __name__ == "__main__":
    generate_pie_charts()
