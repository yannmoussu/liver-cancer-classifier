#!/usr/bin/env python3
"""
Compétition Match-Play : Pseudo-features vs Catalogue Radiomique Global (LASSO)
- Conserve les pseudo-features construites à partir de 'parametres_correlation.csv'.
- Isole toutes les autres features numériques de 'global' n'ayant pas servi aux combinaisons.
- EXCLUSION STRICTE : Retire 'id', 'patient_num', 'Age_at_disease' et 'Gender' des features.
- Concatène les deux mondes et entraîne une régression logistique LASSO (L1).
- Évalue la performance et extrait les survivants pour voir qui l'emporte.
- Ajout : Analyse en Composantes Principales (ACP) et ses graphiques associés.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
import warnings
from validation import evaluer_modele_kfold

warnings.filterwarnings('ignore')

def extract_patient_number(val):
    """Extrait proprement un entier unique depuis n'importe quel format (chaîne ou numérique)"""
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    match = re.search(r'(?:^|/|\\|_| )(\d+)(?:$|/|\\|_| )', val_str)
    if match:
        return int(match.group(1))
    match_any = re.search(r'(\d+)', val_str)
    if match_any:
        return int(match_any.group(1))
    return None

def load_and_merge_data(imagerie_path, descriptif_path):
    """Charge les deux fichiers d'imagerie/clinique et réalise la jointure."""
    df_imagerie = pd.read_csv(imagerie_path, sep=';')
    df_clinical = pd.read_csv(descriptif_path, sep=';')
    
    df_imagerie.columns = df_imagerie.columns.str.strip()
    df_clinical.columns = df_clinical.columns.str.strip()
    
    # On conserve les métadonnées utiles cliniques pour la validation croisée
    df_clinical_clean = df_clinical[['patient_num', 'classe_name', 'Age_at_disease', 'Gender']].drop_duplicates()
    df_merged = pd.merge(df_imagerie, df_clinical_clean, on='patient_num', how='inner')
    
    # Résolution des conflits de noms de colonnes causés par la jointure
    if 'classe_name_x' in df_merged.columns:
        df_merged['classe_name'] = df_merged['classe_name_x']
        df_merged = df_merged.drop(columns=['classe_name_x', 'classe_name_y'], errors='ignore')
    elif 'classe_name_y' in df_merged.columns:
        df_merged['classe_name'] = df_merged['classe_name_y']
        df_merged = df_merged.drop(columns=['classe_name_y'], errors='ignore')
        
    return df_merged

def load_specific_files():
    """Charge les fichiers nécessaires depuis le dossier 'data/' et la racine"""
    base_dir = "data"
    
    path_global = os.path.join(base_dir, "global_excel_resampled_normalized_flattened.csv")
    path_parametres = os.path.join(base_dir, "poids_modeles_radiomiques.csv")
    imagerie_path = 'data/Relectures_imageries.csv'
    descriptif_path = 'data/Descriptif_patients(Sheet1).csv'
    
    for p in [path_global, path_parametres, imagerie_path, descriptif_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERREUR] Fichier introuvable : {p}")
            
    print("[OK] Tous les fichiers sources requis ont été localisés.")
    
    df_glob = pd.read_csv(path_global, sep=';', low_memory=False).rename(columns=lambda x: x.strip())
    df_params = pd.read_csv(path_parametres, sep=';').rename(columns=lambda x: x.strip())
    df_relec = load_and_merge_data(imagerie_path, descriptif_path)
    
    return df_glob, df_params, df_relec

def main():
    try:
        df_glob, df_params, df_relec = load_specific_files()
    except FileNotFoundError as e:
        print(e)
        return

    # Chemin cible pour la sauvegarde des plots demandés
    output_plot_dir = 'plots/optimisés_lasso'
    os.makedirs(output_plot_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. PREPROCESSING DES RELECTURES 
    # -------------------------------------------------------------------------
    df_relec['classe_name'] = df_relec['classe_name'].astype(str).str.strip()
    df_relec = df_relec[df_relec['classe_name'].isin(['CHC', 'CCK'])].copy()
    
    df_relec = df_relec.replace('', np.nan).fillna(0)
    
    # Définition de la cible : 0 pour CCK, 1 pour CHC
    df_relec['target'] = df_relec['classe_name'].map({'CCK': 0, 'CHC': 1})
    
    df_relec['patient_id'] = df_relec['patient_num'].apply(extract_patient_number)
    df_relec = df_relec.dropna(subset=['patient_id']).set_index('patient_id')
    df_relec = df_relec[~df_relec.index.duplicated(keep='first')]

    # -------------------------------------------------------------------------
    # 2. NETTOYAGE DU CATALOGUE RADIOMIQUE (Global)
    # -------------------------------------------------------------------------
    col_patient_glob = 'patient_num' if 'patient_num' in df_glob.columns else df_glob.columns[0]
    df_glob['patient_id'] = df_glob[col_patient_glob].apply(extract_patient_number)
    df_glob = df_glob.dropna(subset=['patient_id']).set_index('patient_id')
    
    df_radio_brut = df_glob.select_dtypes(include=[np.number])
    df_radio_brut = df_radio_brut[~df_radio_brut.index.duplicated(keep='first')]

    # -------------------------------------------------------------------------
    # 3. ALIGNEMENT STRICT DES PATIENTS
    # -------------------------------------------------------------------------
    patients_communs = df_radio_brut.index.intersection(df_relec.index)
    
    if len(patients_communs) == 0:
        print("[ERREUR CRITIQUE] Aucun patient en commun trouvé après alignement.")
        return

    df_radio = df_radio_brut.loc[patients_communs]
    df_target = df_relec.loc[patients_communs]
    
    print(f"[OK] Alignement réussi ! Population commune d'analyse : {len(patients_communs)} patients.")

    # Standardisation des features radiomiques (Z-score)
    scaler_radio = StandardScaler()
    df_radio_scaled = pd.DataFrame(
        scaler_radio.fit_transform(df_radio),
        index=df_radio.index,
        columns=df_radio.columns
    )

    # -------------------------------------------------------------------------
    # 4. RECONSTRUCTION DES PSEUDO-FEATURES LINÉAIRES
    # -------------------------------------------------------------------------
    print("\nCalcul des pseudo-features linéaires à partir des poids de corrélation...")
    df_pseudo_features = pd.DataFrame(index=patients_communs)
    used_features = set()

    for idx, row in df_params.iterrows():
        cible = row['Variable_Clinique']
        intercept = float(row['Intercept_Constante'])
        
        f1, w1 = row['Feature_1'], float(row['Poids_Feature_1'])
        f2, w2 = row['Feature_2'], float(row['Poids_Feature_2'])
        f3, w3 = row['Feature_3'], float(row['Poids_Feature_3'])
        
        if f1 in df_radio_scaled.columns and f2 in df_radio_scaled.columns and f3 in df_radio_scaled.columns:
            z = intercept + (w1 * df_radio_scaled[f1]) + (w2 * df_radio_scaled[f2]) + (w3 * df_radio_scaled[f3])
            df_pseudo_features[f"pseudo_{cible}"] = z
            used_features.update([f1, f2, f3])

    if df_pseudo_features.empty:
        print("[ERREUR] Impossible de construire les variables composites.")
        return

    print(f"[OK] {len(df_pseudo_features.columns)} pseudo-features radiologues générées avec succès.")

    # -------------------------------------------------------------------------
    # 5. ISOLATION DES FEATURES DU CATALOGUE BRUT (Matière première restante)
    # -------------------------------------------------------------------------
    exclude_metadata = ['id', 'patient_num', 'id_x', 'id_y', 'Age_at_disease', 'Gender']
    all_numeric_cols = df_radio_scaled.columns
    
    raw_features_cols = [col for col in all_numeric_cols if col not in used_features and col not in exclude_metadata]
    df_raw_features = df_radio_scaled[raw_features_cols]

    # Concaténation des deux mondes : Pseudo-features vs Features brutes d'imagerie
    X_competition = pd.concat([df_pseudo_features, df_raw_features], axis=1)
    y_final = df_target['target'].astype(int)

    print(f"Taille de la matrice de compétition : {X_competition.shape} ({len(df_pseudo_features.columns)} pseudo-features, {len(raw_features_cols)} features brutes)")

    scaler_final = StandardScaler()
    X_competition_scaled = pd.DataFrame(
        scaler_final.fit_transform(X_competition),
        index=X_competition.index,
        columns=X_competition.columns
    )

    # -------------------------------------------------------------------------
    # 6. ENTRAÎNEMENT DE LA RÉGRESSION LOGISTIQUE LASSO (L1)
    # -------------------------------------------------------------------------
    print("\nEntraînement de la régression logistique pénalisée LASSO (C=0.4)...")
    lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42, max_iter=2000)
    lasso_clf.fit(X_competition_scaled, y_final)

    y_pred = lasso_clf.predict(X_competition_scaled)
    y_probs = lasso_clf.predict_proba(X_competition_scaled)[:, 1]

    print("\n" + "="*80)
    print("   RAPPORT DE PERFORMANCE DE LA COMPÉTITION (MATCH-PLAY LASSO)")
    print("="*80)
    print(classification_report(y_final, y_pred, target_names=['CCk', 'CHC']))
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_final, y_pred))

    # -------------------------------------------------------------------------
    # 7. SÉLECTION DES SURVIVANTS DU LASSO & DESIGN DES GRAPHES
    # -------------------------------------------------------------------------
    coefs = lasso_clf.coef_[0]
    df_coefficients = pd.DataFrame({
        'Feature': X_competition.columns,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    })

    df_survivants = df_coefficients[df_coefficients['Coefficient'] != 0].sort_values(by='Abs_Coefficient', ascending=False)
    print(f"\n[LASSO] Nombre de variables sélectionnées (survivantes) : {len(df_survivants)} / {X_competition.shape[1]}")
    print(df_survivants[['Feature', 'Coefficient']])

    # Sauvegarde des coefficients non nuls
    df_survivants_csv = df_survivants.copy()
    df_survivants_csv['Coefficient'] = df_survivants_csv['Coefficient'].astype(str).str.replace('.', ',')
    df_survivants_csv['Abs_Coefficient'] = df_survivants_csv['Abs_Coefficient'].astype(str).str.replace('.', ',')
    df_survivants_csv.to_csv("data/survivants_lasso_competition.csv", sep=';', index=False)

    # Graphique standard : Histogramme horizontal des coefficients LASSO survivants
    if not df_survivants.empty:
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        
        # Coloration sémantique (Négatif = Oriente CCk [Bleu] | Positif = Oriente CHC [Rouge])
        df_survivants['Color_Group'] = df_survivants['Coefficient'].apply(lambda x: '#d62728' if x >= 0 else '#1f77b4')
        
        sns.barplot(
            data=df_survivants.head(20), 
            x='Coefficient', 
            y='Feature', 
            palette=df_survivants['Color_Group'].head(20).tolist(),
            hue='Feature',
            legend=False
        )
        plt.title("Top variables sélectionnées par le LASSO (Rouge = Oriente CHC | Bleu = Oriente CCk)", fontsize=12, fontweight='bold')
        plt.xlabel("Coefficient dans le modèle")
        plt.ylabel("Nom de la Feature")
        plt.tight_layout()
        plt.savefig(f'{output_plot_dir}/competition_lasso_survivants.png', dpi=300)
        plt.close()
        print(f"[OK] Graphique de la compétition enregistré dans {output_plot_dir}/")

    # -------------------------------------------------------------------------
    # 8. AJOUT : ANALYSE EN COMPOSANTES PRINCIPALES (ACP / PCA)
    # -------------------------------------------------------------------------
    print("\nCalcul et génération des figures de l'ACP...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_competition_scaled)
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'type': y_final.map({0: 'CCk', 1: 'CHC'})
    })
    
    # Graphique PCA 1 : Scatter plot des composantes
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='type', palette='Set1', s=60, alpha=0.7)
    plt.title('PCA of Competition Space (CCk vs CHC)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(f'{output_plot_dir}/pca_chc_cck_pseudo_features.png', dpi=300)
    plt.close()

    # Graphique PCA 2 : Frontière de décision sur espace ACP
    logreg_pca = LogisticRegression(random_state=42, max_iter=1000)
    logreg_pca.fit(X_pca, y_final)
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
    plt.savefig(f'{output_plot_dir}/pca_chc_cck_decision_boundary.png', dpi=300)
    plt.close()

    # Graphique PCA 3 : Scree plot de la variance expliquée
    pca_full = PCA().fit(X_competition_scaled)
    explained_var = pca_full.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    components = range(1, min(len(explained_var) + 1, 30))  # Limité aux 30 premières pour la lisibilité
    plt.bar(components, explained_var[:29], alpha=0.7, color='steelblue')
    plt.step(components, np.cumsum(explained_var[:29]), where='mid', label='Cumulative variance', color='orange', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance (Scree Plot) - Competition Space')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_plot_dir}/pca_explained_variance.png', dpi=300)
    plt.close()
    
    print(f"[OK] Les 3 graphiques ACP ont été enregistrés dans {output_plot_dir}/")

    # -------------------------------------------------------------------------
    # 9. VALIDATION CROISÉE (Stratified Group K-Fold)
    # -------------------------------------------------------------------------
    print("\n=== Évaluation par Cross-Validation (Stratified Group K-Fold) ===")
    y_vrais_cv, y_prob_cv, metrics_cv = evaluer_modele_kfold(
        modele=lasso_clf,
        X=X_competition_scaled, 
        y=y_final.to_numpy(),                                                       
        data=df_target,                                                             
        col_group='patient_num',                                                 
        noms_classes=['CCk', 'CHC'],                                                
        col_age='Age_at_disease' if 'Age_at_disease' in df_target.columns else None, 
        col_sexe='Gender' if 'Gender' in df_target.columns else None,                
        n_splits=5,
        random_seed=42,
        show_plots=False                                                            
    )
    print("\n[SUCCÈS] Script exécuté intégralement.")

if __name__ == '__main__':
    main()