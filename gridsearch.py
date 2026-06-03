import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from validation import *

input_list = [
    'global_excel_resampled_normalized_flattened.csv',
    'global_excel_resampled_normalized_flattened_deltas.csv',
    'global_normalized_flattened_uncor.csv',
    'flattened_global_scaled.csv',
    'flatted_global_scaled_uncor.csv',
    'flatted_global_scaled_uncor_08.csv'
]

C_list = [5, 4, 3, 2, 1, 0.5, 0.3, 0.1]

results = []

for input_file in input_list:
    for C in C_list:
        try:
            df = pd.read_csv("./data/" + input_file, sep=";")
        except FileNotFoundError:
            print(f"Fichier {input_file} introuvable. On l'ignore.")
            continue

        # Suppression des lignes de la classe "Mixtes" pour passer à un problème binaire
        df = df[df["classe_name"] != "Mixtes"]

        # Encodage textuel de la colonne cible ("CCK" et "CHC" -> 0 et 1)
        label_encoder = LabelEncoder()
        df["target"] = label_encoder.fit_transform(df["classe_name"])

        # Sélection de toutes les features (en excluant les identifiants et la cible)
        metadata_cols = ["id", "patient_num", "classe_name", "target"]
        feature_cols = [col for col in df.columns if col not in metadata_cols]

        X = df[feature_cols]
        y = df["target"]

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            random_state=42,
        )

        # On appelle la validation croisée en masquant les graphes individuels
        try:
            y_vrais_total, y_prob_total, fold_metrics = evaluer_modele_kfold(
                model, X, y, df, 
                col_group='id', 
                noms_classes=['CHC', 'CCK'], 
                col_age='Age_at_disease', 
                col_sexe='Gender',
                show_plots=False
            )
            
            results.append({
                'Input': input_file,
                'C': C,
                'Accuracy': np.mean(fold_metrics['accuracy']),
                'AUC': np.mean(fold_metrics['auc']),
                'F1': np.mean(fold_metrics['f1']),
                'MCC': np.mean(fold_metrics['mcc'])
            })
        except Exception as e:
            print(f"Erreur lors de l'évaluation de {input_file} avec C={C} : {e}")

# ==========================================
# GÉNÉRATION DES HEATMAPS GLOBALES
# ==========================================
df_results = pd.DataFrame(results)

if not df_results.empty:
    metrics = ['Accuracy', 'AUC', 'F1', 'MCC']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # Pivot des données pour seaborn (Lignes: Fichiers, Colonnes: C, Valeurs: Métrique)
        pivot_table = df_results.pivot(index='Input', columns='C', values=metric)
        sns.heatmap(pivot_table, annot=True, cmap='viridis', ax=axes[i], fmt='.3f')
        axes[i].set_title(f'Heatmap - {metric}', fontweight='bold')
        axes[i].set_ylabel('Fichier Input')
        axes[i].set_xlabel('Paramètre C (Régularisation)')

    plt.suptitle("Comparaison des métriques (Moyennes k-fold) selon l'Input et C", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Aucun résultat n'a pu être généré. Vérifiez les fichiers d'entrée.")
