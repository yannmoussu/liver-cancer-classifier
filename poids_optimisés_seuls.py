#!/usr/bin/env python3
"""
Méta-Classifieur CCK vs CHC basé sur des index cliniques composites continus.
- Version épurée : utilise uniquement le catalogue de features 'global'.
- S'inspire strictement de la logique de preprocessing de 'modele_radiologue.py' :
  - Cartographie finale : 0 pour CCK, 1 pour CHC.
- Reconstruit les pseudo-features de manière purement linéaire (sans sigmoïde)
  en se basant sur 'parametres_correlation.csv'.
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
    output_plot_dir = 'plots/optimisés_seuls'
    os.makedirs(output_plot_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. PREPROCESSING DES RELECTURES 
    # -------------------------------------------------------------------------
    df_relec['classe_name'] = df_relec['classe_name'].astype(str).str.strip()
    # Filtre strict : Ne garder que CCK et CHC
    df_relec = df_relec[df_relec['classe_name'].isin(['CHC', 'CCK'])].copy()
    
    # Remplacement des vides
    df_relec = df_relec.replace('', np.nan).fillna(0)
    
    # Définition de la cible : 0 pour CCK, 1 pour CHC
    df_relec['target'] = df_relec['classe_name'].map({'CCK': 0, 'CHC': 1})
    
    # Extraction propre de l'identifiant patient numérique
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

    for idx, row in df_params.iterrows():
        cible = row['Variable_Clinique']
        intercept = float(row['Intercept_Constante'])
        
        f1, w1 = row['Feature_1'], float(row['Poids_Feature_1'])
        f2, w2 = row['Feature_2'], float(row['Poids_Feature_2'])
        f3, w3 = row['Feature_3'], float(row['Poids_Feature_3'])
        
        if f1 in df_radio_scaled.columns and f2 in df_radio_scaled.columns and f3 in df_radio_scaled.columns:
            z = intercept + (w1 * df_radio_scaled[f1]) + (w2 * df_radio_scaled[f2]) + (w3 * df_radio_scaled[f3])
            df_pseudo_features[f"pseudo_{cible}"] = z

    if df_pseudo_features.empty:
        print("[ERREUR] Impossible de construire les variables composites.")
        return

    print(f"[OK] {len(df_pseudo_features.columns)} pseudo-features radiologues générées avec succès.")

    # -------------------------------------------------------------------------
    # 5. ENTRAÎNEMENT DU MÉTA-CLASSIFIEUR GLOBAL
    # -------------------------------------------------------------------------
    X_final = df_pseudo_features
    y_final = df_target['target'].astype(int)

    # Standardisation finale des pseudo-features
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    # Entraînement de la Régression Logistique
    meta_clf = LogisticRegression(random_state=42, max_iter=1000)
    meta_clf.fit(X_final_scaled, y_final)

    y_pred = meta_clf.predict(X_final_scaled)
    y_probs = meta_clf.predict_proba(X_final_scaled)[:, 1]

    # -------------------------------------------------------------------------
    # 6. EXPORT DES RÉSULTATS & PERFORMANCE
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("   RAPPORT DE PERFORMANCE DU CLASSIFIEUR COMPOSITE (PSEUDO-RADIOLOGUE)")
    print("="*80)
    print(classification_report(y_final, y_pred, target_names=['CCk', 'CHC']))
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_final, y_pred))

    # Extraction de l'importance des pseudo-features
    coefs = meta_clf.coef_[0]
    df_poids_finaux = pd.DataFrame({
        'Critere_Semiologique': [c.replace('pseudo_', '') for c in X_final.columns],
        'Coefficient_Brut': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values(by='Abs_Coefficient', ascending=False)

    # Export CSV
    df_poids_finaux_csv = df_poids_finaux.copy()
    df_poids_finaux_csv['Coefficient_Brut'] = df_poids_finaux_csv['Coefficient_Brut'].astype(str).str.replace('.', ',')
    df_poids_finaux_csv['Abs_Coefficient'] = df_poids_finaux_csv['Abs_Coefficient'].astype(str).str.replace('.', ',')
    df_poids_finaux_csv.to_csv("data/poids_decisionnels_chc_cck_lineaire.csv", sep=';', index=False)

    # -------------------------------------------------------------------------
    # 7. GRAPHISME STANDARD : COURBE ROC & IMPORTANCE DES COEFFS
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#d62728' if c >= 0 else '#1f77b4' for c in df_poids_finaux['Coefficient_Brut']]
    sns.barplot(data=df_poids_finaux.head(15), x='Coefficient_Brut', y='Critere_Semiologique', ax=ax1, palette=colors, hue='Critere_Semiologique', legend=False)
    ax1.set_title("Importance des critères (Bleu = Oriente CCk | Rouge = Oriente CHC)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Valeur du Coefficient du modèle")
    ax1.set_ylabel("Signe Sémantique Radiologue")
    
    fpr, tpr, _ = roc_curve(y_final, y_probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='#9467bd', lw=3, label=f'Jumeau Numérique (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_title("Capacité globale à distinguer CCk et CHC", fontsize=11, fontweight='bold')
    ax2.set_xlabel("1 - Spécificité (Faux Positifs)")
    ax2.set_ylabel("Sensibilité (Vrais Positifs)")
    ax2.legend(loc="lower right")
    
    plt.suptitle("ARCHITECTURE MÉTA-CLASSIFIEUR COMPOSITE LINÉAIRE (IA EXPLICABLE)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_plot_dir}/performance_classifieur_lineaire.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 8. AJOUT : ANALYSE EN COMPOSANTES PRINCIPALES (ACP / PCA)
    # -------------------------------------------------------------------------
    print("\nCalcul et génération des figures de l'ACP...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_final_scaled)
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'type': y_final.map({0: 'CCk', 1: 'CHC'})
    })
    
    # Graphique PCA 1 : Scatter plot des composantes
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='type', palette='Set1', s=60, alpha=0.7)
    plt.title('PCA of Composite Features (CCk vs CHC)')
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
    pca_full = PCA().fit(X_final_scaled)
    explained_var = pca_full.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    components = range(1, len(explained_var) + 1)
    plt.bar(components, explained_var, alpha=0.7, color='steelblue')
    plt.step(components, np.cumsum(explained_var), where='mid', label='Cumulative variance', color='orange', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance (Scree Plot) - Pseudo Features')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_plot_dir}/pca_explained_variance.png', dpi=300)
    plt.close()
    
    print(f"[OK] Les 3 graphiques ACP ont été enregistrés dans {output_plot_dir}/")

    # -------------------------------------------------------------------------
    # 9. VALIDATION CROISÉE (Stratified Group K-Fold)
    # -------------------------------------------------------------------------
    print("\n=== Cross-validation with Stratified Group K-Fold (by Patient) ===")
    y_vrais_cv, y_prob_cv, metrics_cv = evaluer_modele_kfold(
        modele=meta_clf,
        X=pd.DataFrame(X_final_scaled, columns=X_final.columns, index=X_final.index), 
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