import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from validation import *

"""
input_list = [
    'global_excel_resampled_normalized_flattened.csv',
    'global_excel_resampled_normalized_flattened_deltas.csv',
    'flattened_global_scaled.csv',
    'flatted_global_scaled_uncor.csv',
    'flatted_global_scaled_uncor_08.csv'
]
"""

input_list = [
    'flatted_global_scaled_uncor.csv'
]
"""
input_titles = [
    'Manually Normalized Flattened',
    'Manually Normalized Flattened with Deltas',
    'Normalized Flattened',
    'Normalized Flattened Uncorrelated (0.9)',
    'Normalized Flattened Uncorrelated (0.8)'
]
"""

input_titles = [
    'Normalized Flattened Uncor',
]

C_list = [5, 4, 3, 2, 1, 0.5, 0.3, 0.1]

results = []

for input_file, input_title in zip(input_list, input_titles):
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
                'Input': input_title,
                'C': C,
                'Accuracy': np.mean(fold_metrics['accuracy']),
                # 'AUC': np.mean(fold_metrics['auc']),
                'F1': np.mean(fold_metrics['f1']),
                # 'MCC': np.mean(fold_metrics['mcc'])
            })
        except Exception as e:
            print(f"Erreur lors de l'évaluation de {input_file} avec C={C} : {e}")

# ==========================================
# GÉNÉRATION DES HEATMAPS GLOBALES
# ==========================================
df_results = pd.DataFrame(results)

if not df_results.empty:
    metrics = ['Accuracy', 'F1']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 14), sharex=True)
    axes = np.atleast_1d(axes)

    # Calculer la plage commune pour la barre de couleur
    all_values = pd.concat([
        df_results.pivot(index='Input', columns='C', values=metric).stack()
        for metric in metrics
    ])
    vmin = all_values.min()
    vmax = all_values.max()

    heatmap_im = None
    for ax, metric in zip(axes, metrics):
        pivot_table = df_results.pivot(index='Input', columns='C', values=metric)
        heatmap_im = sns.heatmap(
            pivot_table,
            annot=True,
            annot_kws={'fontsize': 12, 'fontweight': 'bold'},
            cmap='BuPu',
            fmt='.1%',
            square=True,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )
        ax.set_title(f'{metric} - Moyennes k-fold', fontweight='bold', fontsize=18)
        if ax is axes[-1]:
            ax.set_xlabel('Paramètre C (Régularisation)', fontsize=14)
        else:
            ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=12)

    fig.suptitle('Comparaison Accuracy et F1 par Input et paramètre C', fontweight='bold', fontsize=20)
    # Récupère le mappable de la dernière heatmap pour la colorbar
    mappable = axes[-1].collections[0]
    cax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('Valeur (%)', labelpad=10, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.show()
else:
    print("Aucun résultat n'a pu être généré. Vérifiez les fichiers d'entrée.")
