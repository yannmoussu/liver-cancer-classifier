#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

def find_file(filename):
    if os.path.exists(filename): return filename
    potential_path = os.path.join('data', filename)
    if os.path.exists(potential_path): return potential_path
    raise FileNotFoundError(f"Fichier {filename} introuvable.")

def main():
    sns.set_theme(style="whitegrid")
    
    # 1. Chargement des données (Exemple ciblé sur la Capsule)
    try:
        df_patients = pd.read_csv(find_file('matrice_patients_pour_regression.csv'), sep=';', index_col='patient_num')
    except Exception as e:
        print(f"[ERREUR] Échec du chargement : {e}")
        return

    cible = 'Capsule'
    features_x = ['original_shape_Sphericity_TARD', 'original_shape_Sphericity_PORT', 'original_shape_Flatness_TARD']
    
    df_model = df_patients[[cible] + features_x].dropna()
    y = df_model[cible]
    X = df_model[features_x]
    
    # 2. Entraînement et calcul des contributions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(fit_intercept=True, penalty=None)
    model.fit(X_scaled, y)
    
    coefficents = model.coef_[0]
    somme_poids_abs = sum(abs(w) for w in coefficents)
    poids_normalises = [abs(w) / somme_poids_abs for w in coefficents]
    
    # Calcul des probabilités pour la courbe ROC
    y_probs = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # 3. Création de la figure (1 ligne, 2 colonnes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- GRAPHIQUE GAUCHE : Importance des features ---
    # Raccourcir les noms pour le graphique
    features_labels = [f.replace('original_shape_', '') for f in features_x]
    colors = ['#1f77b4' if w >= 0 else '#d62728' for w in coefficents]
    
    sns.barplot(x=poids_normalises, y=features_labels, ax=ax1, palette=colors, hue=features_labels, legend=False)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Poids Normalisé (Part de contribution à la décision)", fontsize=11)
    ax1.set_title(f"Recette du modèle pour imiter le Radiologue ({cible})", fontsize=12, fontweight='bold')
    
    # Ajout des valeurs textuelles sur les barres
    for i, p in enumerate(poids_normalises):
        signe = "(+)" if coefficents[i] >= 0 else "(-)"
        ax1.text(p + 0.02, i, f"{p:.2f} {signe}", va='center', fontweight='bold')
        
    # --- GRAPHIQUE DROITE : Courbe ROC d'imitation ---
    ax2.plot(fpr, tpr, color='#2ca02c', lw=2.5, label=f'Jumeau Numérique (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Taux de faux positifs (1 - Spécificité)", fontsize=11)
    ax2.set_ylabel("Taux de vrais positifs (Sensibilité)", fontsize=11)
    ax2.set_title("Capacité du modèle à reproduire le signe clinique", fontsize=12, fontweight='bold')
    ax2.legend(loc="lower right", frameon=True)
    
    plt.suptitle(f"FOCALISATION SUR LA RÉGRESSION : Exemple du critère '{cible}'", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_png = 'visualisation_explicative_regression.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Graphique explicatif généré avec succès : {output_png}")

if __name__ == '__main__':
    main()