#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

def load_and_process_radiomics(csv_path, sep=';', id_col='patient_num'):
    """Charge le fichier radiomique et prépare les features géométriques et d'intensité."""
    df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    df.columns = df.columns.str.strip()
    
    if id_col not in df.columns:
        raise ValueError(f"Colonne ID '{id_col}' non trouvée dans {csv_path}")
        
    # Séparation des colonnes de forme (Shape)
    shape_cols = [c for c in df.columns if 'shape' in c.lower()]
    df_shape = df[[id_col] + shape_cols].groupby(id_col).mean()

    # Gestion de la cinétique temporelle par ligne du bloc patient
    # Crée un index de 0 à 8 pour les 9 lignes de chaque patient
    df_copy = df.copy()
    df_copy['line_index'] = df_copy.groupby(id_col).cumcount()
    
    # On cherche la colonne principale d'intensité moyenne
    mean_col = 'original_firstorder_Mean'
    if mean_col not in df_copy.columns:
        mean_cols = [c for c in df_copy.columns if 'mean' in c.lower()]
        mean_col = mean_cols[0] if mean_cols else df_copy.select_dtypes(include=[np.number]).columns[0]
        
    # Extraction des 3 premières lignes du bloc du patient
    p1 = df_copy[df_copy['line_index'] == 0][[id_col, mean_col]].rename(columns={mean_col: 'Ligne_Bloc_0'}).set_index(id_col)
    p2 = df_copy[df_copy['line_index'] == 1][[id_col, mean_col]].rename(columns={mean_col: 'Ligne_Bloc_1'}).set_index(id_col)
    p3 = df_copy[df_copy['line_index'] == 2][[id_col, mean_col]].rename(columns={mean_col: 'Ligne_Bloc_2'}).set_index(id_col)
    
    df_kinetics = pd.concat([p1, p2, p3], axis=1)
    
    # Calcul des Deltas basés sur les lignes du fichier
    df_kinetics['Delta_Ligne_1_0'] = df_kinetics['Ligne_Bloc_1'] - df_kinetics['Ligne_Bloc_0']
    df_kinetics['Delta_Ligne_2_1'] = df_kinetics['Ligne_Bloc_2'] - df_kinetics['Ligne_Bloc_1']

    # Fusion finale (1 ligne par patient)
    df_processed = pd.concat([df_shape, df_kinetics], axis=1)
    return df_processed

def main():
    global_path = 'data/global_excel_resampled_scaled.csv'
    relectures_path = 'data/Relectures_imageries(Feuil1).csv'

    print("1. Traitement du fichier global...")
    global_df = load_and_process_radiomics(global_path)

    print("2. Chargement du fichier Relectures...")
    relectures_raw = pd.read_csv(relectures_path, sep=';')
    relectures_raw.columns = relectures_raw.columns.str.strip()
    
    if 'Patient_number' not in relectures_raw.columns:
        raise ValueError("Patient_number non trouvé dans le fichier Relectures")
        
    # Conversion forcée des cibles cliniques en valeurs numériques (0 ou 1)
    cols_cliniques = ['Size_mm', 'Shape', 'nonrim_APHE', 'Nonperiph_washout', 'Necrosis', 'APHE_heterogeneous']
    for col in cols_cliniques:
        if col in relectures_raw.columns:
            relectures_raw[col] = pd.to_numeric(relectures_raw[col], errors='coerce').fillna(0)

    relectures_df = relectures_raw[['Patient_number'] + [c for c in cols_cliniques if c in relectures_raw.columns]].set_index('Patient_number')
    
    # Alignement des patients communs
    common_idx = global_df.index.intersection(relectures_df.index)
    global_df = global_df.loc[common_idx]
    relectures_df = relectures_df.loc[common_idx]

    # Calcul des corrélations de Spearman
    records = []
    for src_col in global_df.columns:
        for tgt_col in relectures_df.columns:
            corr_val = global_df[src_col].corr(relectures_df[tgt_col], method='spearman')
            if not np.isnan(corr_val):
                records.append({
                    'source_feature': src_col,
                    'target_feature': tgt_col,
                    'correlation': float(corr_val),
                    'abs_correlation': abs(float(corr_val))
                })

    records.sort(key=lambda x: x['abs_correlation'], reverse=True)
    out_df = pd.DataFrame(records)
    
    # Sauvegarde
    output_path = 'correlations_global_multislice_relectures.csv'
    out_df.to_csv(output_path, index=False)
    print(f"\n[Succès] Fichier enregistré : {output_path}")
    
    # Affichage du Top 15 pour contrôle
    print("\n--- TOP 15 DES CORRÉLATIONS ---")
    print(out_df.head(15).to_string(index=False))

if __name__ == '__main__':
    main()