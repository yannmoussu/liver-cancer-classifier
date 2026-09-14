#!/usr/bin/env python3
"""
Modele de classification des tumeurs basé sur les evaluations radiologues.
Objectif: classification dure (logistic regression) entre CCk (0) et CHC (1)
à partir des fichiers Relectures_imageries.csv et Descriptif_patients(Sheet1).csv.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.validation import evaluer_modele_kfold

def load_and_merge_data(imagerie_path, descriptif_path):
    """Charge les deux fichiers et réalise la jointure sur le numéro de patient."""
    df_imagerie = pd.read_csv(imagerie_path, sep=';')
    df_clinical = pd.read_csv(descriptif_path, sep=';')
    
    # On nettoie les espaces ou retours à la ligne masqués dans les noms de colonnes
    df_imagerie.columns = df_imagerie.columns.str.strip()
    df_clinical.columns = df_clinical.columns.str.strip()
    
    # On extrait uniquement les colonnes nécessaires du descriptif clinique
    df_clinical_clean = df_clinical[['patient_num', 'classe_name']].drop_duplicates()
    
    # Jointure sur 'patient_num'
    df_merged = pd.merge(df_imagerie, df_clinical_clean, on='patient_num', how='inner')
    
    # AJOUT RECOURS : Si Pandas a créé des suffixes à cause du doublon, on répare
    if 'classe_name_x' in df_merged.columns:
        df_merged['classe_name'] = df_merged['classe_name_x']
        df_merged = df_merged.drop(columns=['classe_name_x', 'classe_name_y'], errors='ignore')
    elif 'classe_name_y' in df_merged.columns:
        df_merged['classe_name'] = df_merged['classe_name_y']
        df_merged = df_merged.drop(columns=['classe_name_y'], errors='ignore')
        
    return df_merged

def preprocess(df):
    """Filtre les classes, définit la cible y et nettoie les caractéristiques X."""
    # On s'assure que les chaînes de texte n'ont pas d'espaces masqués (ex: "CHC " au lieu de "CHC")
    df['classe_name'] = df['classe_name'].astype(str).str.strip()
    
    # On ne garde que les CHC et les CCK
    df = df[df['classe_name'].isin(['CHC', 'CCK'])].copy()
    
    # Définition de la cible (0 pour CCK, 1 pour CHC)
    y = df['classe_name'].map({'CCK': 0, 'CHC': 1})
    
    # Liste des caractéristiques (on retire absolument tout identifiant clinique)
    exclude_cols = ['classe_name', 'patient_num', 'id', 'classe_name_x', 'classe_name_y']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    X = X.replace('', np.nan)
    X = X.fillna(0)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, y, feature_cols, df

def main():
    imagerie_path = 'data/Relectures_imageries.csv'
    descriptif_path = 'data/Descriptif_patients(Sheet1).csv'
    
    # Chargement et fusion des données
    df_raw = load_and_merge_data(imagerie_path, descriptif_path)
    print(f"Dataset shape après fusion: {df_raw.shape}")
    print(f"Distribution initiale des tumeurs:\n{df_raw['classe_name'].value_counts()}")

    # Preprocessing
    X, y, feature_names, df = preprocess(df_raw)
    print(f"Features shape: {X.shape}")
    print(f"Distribution après filtrage (0: CCK, 1: CHC):\n{y.value_counts()}")

    # Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Régression Logistique
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train_scaled, y_train)

    # Prédictions
    y_pred = logreg.predict(X_test_scaled)

    print("\n=== Logistic Regression Results ===")
    print("Accuracy:", logreg.score(X_test_scaled, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CCk', 'CHC']))
    
    # Calcul de la matrice de confusion pour le Split de Test
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Génération de la Heatmap de la matrice de confusion
    plt.figure(figsize=(6, 5))
    with plt.rc_context({
        'font.weight': 'bold', 
        'font.size': 16,
        'axes.labelweight': 'bold'
    }):
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=['CCk', 'CHC'], 
            yticklabels=['CCk', 'CHC'], 
            cbar=False,
            annot_kws={"size": 48, "weight": "bold"}
        )
        for text in ax.texts:
            text.set_size(18)          
            text.set_weight('bold')    
            text.set_color('black')    

    plt.title('Matrice de Confusion (Test Split)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Vérité Terrain', fontsize=12, fontweight='bold')
    plt.xlabel('Prédiction Modèle', fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/confusion_matrix_test_split.png', dpi=300)
    plt.close()

    # Importance des variables (Coefficients)
    coefs = logreg.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs)
    }).sort_values('abs_coefficient', ascending=False)

    print("\n=== Feature Importance (Top 15) ===")
    print(feature_importance[['feature', 'coefficient']].head(15))

    # Graphique Importance des variables
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(15), x='coefficient', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Top 15 Features - Logistic Regression Coefficients')
    plt.xlabel('Coefficient')
    plt.tight_layout()
    plt.savefig('plots/logreg_feature_importance_Relectures_imagerie.png')
    plt.close()

    # Analyse en Composantes Principales (ACP / PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'type': y_train.map({0: 'CCk', 1: 'CHC'})
    })
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='type', palette='Set1', s=60, alpha=0.7)
    plt.title('PCA of Training Set (CCk vs CHC)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig('plots/pca_chc_cck_Relectures_imagerie.png')
    plt.close()

    # PCA avec frontière de décision
    logreg_pca = LogisticRegression(random_state=42, max_iter=1000)
    logreg_pca.fit(X_pca, y_train)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = logreg_pca.predict_proba(grid)[:, 1].reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs, levels=25, cmap='RdBu', alpha=0.3)
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='type', palette='Set1', s=60, alpha=0.8, edgecolor='k')
    plt.title('PCA with Logistic Regression Decision Boundary (CCk vs CHC)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig('plots/pca_chc_cck_decision_boundary_Relectures_imagerie.png')
    plt.close()

    # Scree Plot (PCA Variance)
    pca_full = PCA().fit(X_train_scaled)
    explained_var = pca_full.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    components = range(1, len(explained_var) + 1)
    plt.bar(components, explained_var, alpha=0.7, color='steelblue')
    plt.step(components, np.cumsum(explained_var), where='mid', label='Cumulative variance', color='orange', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance (Scree Plot)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('plots/pca_explained_variance_Relectures_imagerie.png')
    plt.close()

    # Matrice de corrélation
    X_df = pd.DataFrame(X, columns=feature_names)
    corr = X_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix_Relectures_imagerie.png')
    plt.close()

    # Validation croisée Stratified Group K-Fold
    print("\n=== Cross-validation with Stratified Group K-Fold (by Patient) ===")
    evaluer_modele_kfold(
        logreg,
        X,
        y,
        df,  
        ['patient_num'],  
        ['CCk', 'CHC']
    )

    # Sauvegarde des artefacts
    import joblib
    joblib.dump(logreg, 'logreg_chc_cck.pkl')
    joblib.dump(scaler, 'scaler_chc_cck.pkl')
    print("\nModèle et scaler sauvegardés avec succès.")

if __name__ == '__main__':
    main()