#!/usr/bin/env python3
"""
Méta-Classifieur CHC vs CCK basé sur des index cliniques composites continus.
- Version épurée : utilise uniquement le catalogue de features 'global'.
- S'inspire strictement de la logique de preprocessing de 'modele_radiologue.py' :
  - CHC codé par 1, CCk codé par 2.
  - Cartographie finale : 0 pour CHC, 1 pour CCk.
- Reconstruit les pseudo-features de manière purement linéaire (sans sigmoïde)
  en se basant sur 'parametres_correlation.csv'.
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
    # Filtre strict : Ne garder que CHC (1) et CCk (2)
    df_relec = df_relec[df_relec['Type_tumeur'].isin([1, 2])].copy()
    
    # Remplacement des vides et conversion numérique pour éviter les bugs
    df_relec = df_relec.replace('', np.nan).fillna(0)
    
    # Définition de la cible : 0 pour CHC, 1 pour CCk
    df_relec['target'] = df_relec['Type_tumeur'].map({1: 0, 2: 1})
    
    # Extraction propre de l'identifiant patient numérique
    df_relec['patient_id'] = df_relec['Patient_number'].apply(extract_patient_number)
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
        print(f"Exemples IDs fichier Global: {list(df_radio_brut.index[:5])}")
        print(f"Exemples IDs fichier Relectures: {list(df_relec.index[:5])}")
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
            # Combinaison linéaire pure continue
            z = intercept + (w1 * df_radio_scaled[f1]) + (w2 * df_radio_scaled[f2]) + (w3 * df_radio_scaled[f3])
            df_pseudo_features[f"pseudo_{cible}"] = z

    if df_pseudo_features.empty:
        print("[ERREUR] Impossible de construire les variables composites. Vérifiez la correspondance des noms de colonnes.")
        return

    print(f"[OK] {len(df_pseudo_features.columns)} pseudo-features radiologues générées avec succès.")

    # -------------------------------------------------------------------------
    # 5. ENTRAÎNEMENT DU MÉTA-CLASSIFIEUR GLOBAL
    # -------------------------------------------------------------------------
    X_final = df_pseudo_features
    y_final = df_target['target'].astype(int)

    # Standardisation finale pour équilibrer l'importance des pseudo-features
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    # Entraînement de la Régression Logistique sur l'ensemble de la population commune
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
    print(classification_report(y_final, y_pred, target_names=['CHC', 'CCk']))
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_final, y_pred))

    # Extraction de l'importance des pseudo-features
    coefs = meta_clf.coef_[0]
    df_poids_finaux = pd.DataFrame({
        'Critere_Semiologique': [c.replace('pseudo_', '') for c in X_final.columns],
        'Coefficient_Brut': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values(by='Abs_Coefficient', ascending=False)

    # Export CSV au standard Excel (séparateur point-virgule, virgule pour les décimales)
    df_poids_finaux_csv = df_poids_finaux.copy()
    df_poids_finaux_csv['Coefficient_Brut'] = df_poids_finaux_csv['Coefficient_Brut'].astype(str).str.replace('.', ',')
    df_poids_finaux_csv['Abs_Coefficient'] = df_poids_finaux_csv['Abs_Coefficient'].astype(str).str.replace('.', ',')
    
    os.makedirs('data', exist_ok=True)
    df_poids_finaux_csv.to_csv("data/poids_decisionnels_chc_cck_lineaire.csv", sep=';', index=False)
    print("\n[OK] Matrice de décision enregistrée : data/poids_decisionnels_chc_cck_lineaire.csv")

    # -------------------------------------------------------------------------
    # 7. GRAPHISME : COURBE ROC & IMPORTANCE DES COEFFS
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1 : Importance des caractéristiques cliniques simulées
    # Un coefficient positif oriente vers le CCk (1), négatif vers le CHC (0)
    colors = ['#d62728' if c >= 0 else '#1f77b4' for c in df_poids_finaux['Coefficient_Brut']]
    sns.barplot(data=df_poids_finaux.head(15), x='Coefficient_Brut', y='Critere_Semiologique', ax=ax1, palette=colors, hue='Critere_Semiologique', legend=False)
    ax1.set_title("Importance des critères (Bleu = Oriente CHC | Rouge = Oriente CCk)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Valeur du Coefficient du modèle")
    ax1.set_ylabel("Signe Sémantique Radiologue")
    
    # Graphique 2 : Courbe ROC
    fpr, tpr, _ = roc_curve(y_final, y_probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='#9467bd', lw=3, label=f'Jumeau Numérique (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_title("Capacité globale à distinguer CHC et CCk", fontsize=11, fontweight='bold')
    ax2.set_xlabel("1 - Spécificité (Faux Positifs)")
    ax2.set_ylabel("Sensibilité (Vrais Positifs)")
    ax2.legend(loc="lower right")
    
    plt.suptitle("ARCHITECTURE MÉTA-CLASSIFIEUR COMPOSITE LINÉAIRE (IA EXPLICABLE)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/performance_classifieur_lineaire.png', dpi=300, bbox_inches='tight')
    print("[OK] Graphique de performance enregistré : plots/performance_classifieur_lineaire.png")
    print("\n[SUCCÈS] Script exécuté intégralement.")

    y_vrais_cv, y_prob_cv, metrics_cv = evaluer_modele_kfold(
        modele=meta_clf,
        X=pd.DataFrame(X_final_scaled, columns=X_final.columns, index=X_final.index), # X_final mis à l'échelle
        y=y_final.to_numpy(),                                                       # Cible (array numpy)
        data=df_target,                                                             # DataFrame contenant les métadonnées
        col_group='Patient_number',                                                 # Regroupement par patient unique
        noms_classes=['CHC', 'CCk'],                                                # Labels des classes (0=CHC, 1=CCk)
        col_age='Age_at_disease' if 'Age_at_disease' in df_target.columns else None, # Ajustement Âge
        col_sexe='Gender' if 'Gender' in df_target.columns else None,                # Ajustement Sexe
        n_splits=5,
        random_seed=42,
        show_plots=False                                                            # Mettre à True si vous exécutez dans un Notebook Jupyter
    )
if __name__ == '__main__':
    main()