#!/usr/bin/env python3
"""
Compétition Match-Play : Pseudo-features vs Catalogue Radiomique Global (LASSO)
- Conserve les pseudo-features construites à partir de 'parametres_correlation.csv'.
- Isole toutes les autres features numériques de 'global' n'ayant pas servi aux combinaisons.
- EXCLUSION STRICTE : Retire 'id', 'patient_num', 'Age_at_disease' et 'Gender' des features.
- Concatène les deux mondes et entraîne une régression logistique LASSO (L1).
- Évalue la performance et extrait les survivants pour voir qui l'emporte.
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

def load_specific_files():
    """Charge les 3 fichiers nécessaires depuis le dossier 'data/'"""
    base_dir = "data"
    
    path_global = os.path.join(base_dir, "global_excel_resampled_normalized_flattened.csv")
    path_parametres = os.path.join(base_dir, "parametres_correlation.csv")
    path_relectures = os.path.join(base_dir, "Relectures_imageries(Feuil1).csv")
    
    for p in [path_global, path_parametres, path_relectures]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERREUR] Fichier introuvable : {p}")
            
    print("[OK] Tous les fichiers sources requis ont été localisés dans data/")
    
    df_glob = pd.read_csv(path_global, sep=';', low_memory=False).rename(columns=lambda x: x.strip())
    df_params = pd.read_csv(path_parametres, sep=';').rename(columns=lambda x: x.strip())
    df_relec = pd.read_csv(path_relectures, sep=';').rename(columns=lambda x: x.strip())
    
    return df_glob, df_params, df_relec

def main():
    try:
        df_glob, df_params, df_relec = load_specific_files()
    except FileNotFoundError as e:
        print(e)
        return

    # -------------------------------------------------------------------------
    # 1. PREPROCESSING DES RELECTURES (Calqué sur modele_radiologue.py)
    # -------------------------------------------------------------------------
    df_relec = df_relec[df_relec['Type_tumeur'].isin([1, 2])].copy()
    df_relec = df_relec.replace('', np.nan).fillna(0)
    df_relec['target'] = df_relec['Type_tumeur'].map({1: 0, 2: 1}) # 0=CHC, 1=CCk
    
    df_relec['patient_id'] = df_relec['Patient_number'].apply(extract_patient_number)
    df_relec = df_relec.dropna(subset=['patient_id']).set_index('patient_id')
    df_relec = df_relec[~df_relec.index.duplicated(keep='first')]

    # -------------------------------------------------------------------------
    # 2. NETTOYAGE DU CATALOGUE RADIOMIQUE (Global)
    # -------------------------------------------------------------------------
    col_patient_glob = 'patient_num' if 'patient_num' in df_glob.columns else df_glob.columns[0]
    df_glob['patient_id'] = df_glob[col_patient_glob].apply(extract_patient_number)
    df_glob = df_glob.dropna(subset=['patient_id']).set_index('patient_id')
    
    # Conservation exclusive des colonnes numériques pour le calcul mathématique
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

    # Standardisation initiale obligatoire du catalogue global (Z-score)
    scaler_radio = StandardScaler()
    df_radio_scaled = pd.DataFrame(
        scaler_radio.fit_transform(df_radio),
        index=df_radio.index,
        columns=df_radio.columns
    )

    # -------------------------------------------------------------------------
    # 4. RECONSTRUCTION DES PSEUDO-FEATURES ET FILTRAGE DES CONCURRENTS
    # -------------------------------------------------------------------------
    print("\nCalcul des pseudo-features linéaires...")
    df_pseudo_features = pd.DataFrame(index=patients_communs)
    features_utilisees = set()

    for idx, row in df_params.iterrows():
        cible = row['Variable_Clinique']
        intercept = float(row['Intercept_Constante'])
        
        f1, w1 = row['Feature_1'], float(row['Poids_Feature_1'])
        f2, w2 = row['Feature_2'], float(row['Poids_Feature_2'])
        f3, w3 = row['Feature_3'], float(row['Poids_Feature_3'])
        
        if f1 in df_radio_scaled.columns and f2 in df_radio_scaled.columns and f3 in df_radio_scaled.columns:
            z = intercept + (w1 * df_radio_scaled[f1]) + (w2 * df_radio_scaled[f2]) + (w3 * df_radio_scaled[f3])
            df_pseudo_features[f"pseudo_{cible}"] = z
            
            # On mémorise les colonnes d'origine à exclure
            features_utilisees.update([f1, f2, f3])

    if df_pseudo_features.empty:
        print("[ERREUR] Impossible de construire les variables composites.")
        return

    # CORRECTION : Ajout explicite de 'id' dans la liste d'exclusion stricte
    metadonnees_a_exclure = ['id', 'patient_num', 'Age_at_disease', 'Gender', 'target', 'patient_id', 'classe_name']
    
    autres_features_cols = [
        col for col in df_radio_scaled.columns 
        if col not in features_utilisees and col not in metadonnees_a_exclure
    ]
    df_global_exclusif = df_radio_scaled[autres_features_cols]

    print(f"[INFO] Compétiteurs prêts : {len(df_pseudo_features.columns)} Pseudo-features VS {len(df_global_exclusif.columns)} Features globales natives.")

    # Fusion des deux matrices pour créer l'espace de compétition complet
    X_competition = pd.concat([df_pseudo_features, df_global_exclusif], axis=1)
    y_final = df_target['target'].astype(int)

    # Standardisation finale de l'ensemble combiné
    scaler_competition = StandardScaler()
    X_competition_scaled = pd.DataFrame(
        scaler_competition.fit_transform(X_competition),
        index=X_competition.index,
        columns=X_competition.columns
    )

    # -------------------------------------------------------------------------
    # 5. ENTRAÎNEMENT DU CLASSIFIEUR LASSO (L1)
    # -------------------------------------------------------------------------
    lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.3, random_state=42, max_iter=2000)
    lasso_clf.fit(X_competition_scaled, y_final)

    y_pred = lasso_clf.predict(X_competition_scaled)
    y_probs = lasso_clf.predict_proba(X_competition_scaled)[:, 1]

    # -------------------------------------------------------------------------
    # 6. EXPORT ET ANALYSE DE LA SÉLECTION DES FEATURES
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("   RAPPORT DE PERFORMANCE DU MATCH COMPETITION (LASSO REGRESSION)")
    print("="*80)
    print(classification_report(y_final, y_pred, target_names=['CHC', 'CCk']))

    coefs = lasso_clf.coef_[0]
    df_coefficients = pd.DataFrame({
        'Feature': X_competition.columns,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs),
        'Type': ['Pseudo-Feature' if c.startswith('pseudo_') else 'Radiomique Brute' for c in X_competition.columns]
    }).sort_values(by='Abs_Coefficient', ascending=False)

    # Filtrer uniquement les variables ayant survécu au LASSO (coefficient différent de 0)
    df_survivants = df_coefficients[df_coefficients['Coefficient'] != 0]
    
    print(f"\n📊 Résultat du tri sélectif Lasso : {len(df_survivants)} features conservées sur {len(X_competition.columns)}.")
    print("\n=== TOP 15 DES FEATURES AYANT SURVÉCU ET LEUR ORIGINE ===")
    print(df_survivants.head(15)[['Feature', 'Type', 'Coefficient']])

    # Exportation Excel-friendly
    df_export = df_survivants.copy()
    df_export['Coefficient'] = df_export['Coefficient'].astype(str).str.replace('.', ',')
    df_export['Abs_Coefficient'] = df_export['Abs_Coefficient'].astype(str).str.replace('.', ',')
    os.makedirs('data', exist_ok=True)
    df_export.to_csv("data/survivants_lasso_competition.csv", sep=';', index=False)
    print("\n[OK] Fichier des survivants sauvegardé : data/survivants_lasso_competition.csv")

    # -------------------------------------------------------------------------
    # 7. GRAPHISME : LES SURVIVANTS DU DUEL
    # -------------------------------------------------------------------------
    if not df_survivants.empty:
        plt.figure(figsize=(11, 7))
        sns.set_theme(style="whitegrid")
        
        palette_dict = {'Pseudo-Feature': '#ff7f0e', 'Radiomique Brute': '#1f77b4'}
        
        sns.barplot(
            data=df_survivants.head(20), 
            x='Coefficient', 
            y='Feature', 
            hue='Type', 
            palette=palette_dict,
            dodge=False
        )
        plt.title("Top des caractéristiques survivantes au LASSO\n(Orange = Vos variables sémiologiques | Bleu = Pixels bruts)", fontsize=12, fontweight='bold')
        plt.xlabel("Coefficient dans le modèle (Négatif = CHC | Positif = CCk)")
        plt.ylabel("Nom de la Feature")
        plt.tight_layout()
        
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/competition_lasso_survivants.png', dpi=300)
        print("[OK] Graphique de la compétition enregistré : plots/competition_lasso_survivants.png")

    # -------------------------------------------------------------------------
    # 8. VALIDATION CROISÉE (Stratified Group K-Fold)
    # -------------------------------------------------------------------------
    print("\n=== Évaluation par Cross-Validation (Stratified Group K-Fold) ===")
    evaluer_modele_kfold(
        modele=lasso_clf,
        X=X_competition_scaled, 
        y=y_final.to_numpy(),                                                       
        data=df_target,                                                             
        col_group='Patient_number',                                                 
        noms_classes=['CHC', 'CCk'],                                                
        col_age='Age_at_disease' if 'Age_at_disease' in df_target.columns else None, 
        col_sexe='Gender' if 'Gender' in df_target.columns else None,                
        n_splits=5,
        random_seed=42,
        show_plots=False                                                            
    )

if __name__ == '__main__':
    main()