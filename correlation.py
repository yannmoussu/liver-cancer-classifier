#!/usr/bin/env python3
"""
Analyse statistique avancée pour le Benchmark Radiomique vs Clinique.
Calcule pour chaque couple (Feature, Cible Clinique) :
- La corrélation de Spearman
- La p-value de Kruskal-Wallis (différence significative entre groupes)
- L'AUC-ROC (capacité discriminante pour les cibles binaires)
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score
import os
import warnings

# Désactiver les avertissements de conversion de type de sklearn
warnings.filterwarnings('ignore')

def find_file(filename):
    """Cherche le fichier à la racine ou dans un sous-dossier 'data'"""
    if os.path.exists(filename):
        return filename
    potential_path = os.path.join('data', filename)
    if os.path.exists(potential_path):
        return potential_path
    raise FileNotFoundError(
        f"\n[ERREUR] Impossible de trouver le fichier '{filename}'.\n"
        f"Assure-toi de l'avoir bien placé dans ton dossier de projet :\n"
        f"-> {os.getcwd()} ou dans {os.getcwd()}/data/"
    )

def main():
    # 1. Détection automatique des fichiers
    try:
        flattened_path = find_file('global_excel_resampled_normalized_flattened_deltas.csv')
        relectures_path = find_file('Relectures_imageries(Feuil1).csv')
    except FileNotFoundError as e:
        print(e)
        return

    print(f"[OK] Fichier radiomique détecté : {flattened_path}")
    print(f"[OK] Fichier clinique détecté   : {relectures_path}\n")
    
    print("1. Chargement du fichier radiomique...")
    df_radio = pd.read_csv(flattened_path, sep=';', low_memory=False)
    df_radio.columns = df_radio.columns.str.strip()
    
    if 'patient_num' not in df_radio.columns:
        raise ValueError("La colonne 'patient_num' est introuvable dans le fichier radiomique.")
    df_radio = df_radio.set_index('patient_num')
    df_radio = df_radio[~df_radio.index.duplicated(keep='first')]

    print("2. Chargement du fichier clinique...")
    df_clinique_raw = pd.read_csv(relectures_path, sep=';')
    df_clinique_raw.columns = df_clinique_raw.columns.str.strip()
    
    if 'Patient_number' not in df_clinique_raw.columns:
        raise ValueError("La colonne 'Patient_number' est introuvable dans le fichier Relectures.")
        
    # Liste des cibles cliniques à analyser
    cols_cliniques = [
        'Size_mm', 'Shape', 'nonrim_APHE', 'Nonperiph_washout', 
        'Necrosis', 'APHE_heterogeneous', 'Late_enhancement', 
        'Satellite_nodule', 'Portal_thrombosis'
    ]
    
    # Conversion numérique forcée des cibles cliniques
    for col in cols_cliniques:
        if col in df_clinique_raw.columns:
            df_clinique_raw[col] = pd.to_numeric(df_clinique_raw[col], errors='coerce').fillna(0)

    df_clinique = df_clinique_raw[['Patient_number'] + [c for c in cols_cliniques if c in df_clinique_raw.columns]].set_index('Patient_number')
    df_clinique = df_clinique[~df_clinique.index.duplicated(keep='first')]

    # 3. Alignement des patients (Intersection stricte)
    common_patients = df_radio.index.intersection(df_clinique.index)
    print(f"-> Patients correspondants trouvés : {len(common_patients)}")
    
    if len(common_patients) == 0:
        print("[ATTENTION] Aucun patient en commun trouvé. Vérifie les colonnes d'identifiants.")
        return
        
    df_radio = df_radio.loc[common_patients]
    df_clinique = df_clinique.loc[common_patients]

    # Isoler les colonnes numériques radiomiques (Deltas inclus)
    radio_numeric_cols = df_radio.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in radio_numeric_cols: radio_numeric_cols.remove('id')

    # 4. Calculs Statistiques
    print("\nCalcul des statistiques (Spearman, Kruskal-Wallis, AUC-ROC)...")
    records = []
    
    for src_col in radio_numeric_cols:
        x = df_radio[src_col]
        if x.nunique() <= 1:
            continue
            
        for tgt_col in df_clinique.columns:
            y = df_clinique[tgt_col]
            
            # --- A. Corrélation de Spearman ---
            spearman_corr = x.corr(y, method='spearman')
            
            # --- B. Test de Kruskal-Wallis ---
            groups = [x[y == val].values for val in y.unique() if len(x[y == val]) > 0]
            if len(groups) > 1:
                try:
                    _, kruskal_p = kruskal(*groups)
                except Exception:
                    kruskal_p = np.nan
            else:
                kruskal_p = np.nan
                
            # --- C. Analyse ROC (AUC) ---
            if y.nunique() == 2 and set(y.unique()).issubset({0, 1}) and len(np.unique(y)) == 2:
                try:
                    x_clean = x.fillna(x.mean())
                    auc_val = roc_auc_score(y, x_clean)
                    abs_auc_effect = abs(auc_val - 0.5) + 0.5
                except Exception:
                    auc_val = np.nan
                    abs_auc_effect = np.nan
            else:
                auc_val = np.nan
                abs_auc_effect = np.nan

            if not np.isnan(spearman_corr):
                records.append({
                    'source_feature': src_col,
                    'target_feature': tgt_col,
                    'spearman_correlation': float(spearman_corr),
                    'abs_spearman': abs(float(spearman_corr)),
                    'kruskal_p_value': float(kruskal_p) if not np.isnan(kruskal_p) else None,
                    'auc_roc': float(auc_val) if not np.isnan(auc_val) else None,
                    'auc_discriminative_strength': float(abs_auc_effect) if not np.isnan(abs_auc_effect) else None
                })

    # Tri global principal par la valeur absolue de Spearman
    records.sort(key=lambda x: x['abs_spearman'], reverse=True)
    df_results = pd.DataFrame(records)

    cols_order = [
        'source_feature', 'target_feature', 'spearman_correlation', 
        'kruskal_p_value', 'auc_roc', 'auc_discriminative_strength'
    ]
    df_results = df_results[cols_order]

    # 5. Sauvegarde
    output_path = 'analyses_avancees_deltas_relectures.csv'
    df_results.to_csv(output_path, index=False, sep=';')
    print(f"[Succès] Matrice statistique sauvegardée dans : {output_path}")

    # 6. Affichage du TOP 15 Global
    print("\n" + "="*85)
    print("TOP 15 DES VARIABLES SELON LEUR FORCE DE CORRÉLATION (SPEARMAN)")
    print("="*85)
    print(df_results.head(15).to_string(index=False, formatters={
        'spearman_correlation': '{:,.4f}'.format,
        'kruskal_p_value': '{:,.4e}'.format,
        'auc_roc': '{:,.4f}'.format,
        'auc_discriminative_strength': '{:,.4f}'.format
    }))

    # 7. Zoom ciblé sur le comportement des DELTAS et des PHASES
    print("\n" + "="*85)
    print("ZOOM STATISTIQUE SUR LES DELTAS CINÉTIQUES ET LES PHASES TEMPORELLES")
    print("="*85)
    df_deltas = df_results[df_results['source_feature'].str.contains('delta|_ART|_PORT|_NAT', case=False, na=False)]
    
    # Pour le zoom cinétique, on trie par p-value de Kruskal croissante (plus petit = plus significatif)
    df_deltas_sorted = df_deltas.sort_value(by='kruskal_p_value', ascending=True)
    
    if not df_deltas_sorted.empty:
        print(df_deltas_sorted.head(20).to_string(index=False, formatters={
            'spearman_correlation': '{:,.4f}'.format,
            'kruskal_p_value': '{:,.4e}'.format,
            'auc_roc': '{:,.4f}'.format,
            'auc_discriminative_strength': '{:,.4f}'.format
        }))
    else:
        print("Aucune feature contenant 'delta', '_ART', '_PORT' ou '_NAT' trouvée.")

if __name__ == '__main__':
    main()