#!/usr/bin/env python3
import pandas as pd

def main():
    # Charger ton fichier de résultats
    results_path = 'data/analyses_avancees_deltas_relectures.csv'
    df = pd.read_csv(results_path, sep=';')
    
    # Nettoyer les espaces s'il y en a
    df['target_feature'] = df['target_feature'].str.strip()
    
    print(f"Total des couples analysés : {len(df)}")
    
    # 1. Filtre pour la Taille (Size_mm) : Spearman fort et p-value significative
    filtre_taille = (df['target_feature'] == 'Size_mm') & \
                     (df['spearman_correlation'].abs() >= 0.80) & \
                     (df['kruskal_p_value'] < 0.05)
                     
    # 2. Filtre pour les critères cliniques qualitatifs : Force AUC >= 0.65 et p-value < 0.05
    filtre_clinique = (df['target_feature'] != 'Size_mm') & \
                       (df['auc_discriminative_strength'] >= 0.65) & \
                       (df['kruskal_p_value'] < 0.05)
                       
    # Combiner les deux filtres
    df_selection = df[filtre_taille | filtre_clinique].copy()
    
    # Trier par cible clinique puis par force de corrélation/discrimination
    df_selection['tri_force'] = df_selection['auc_discriminative_strength'].fillna(df_selection['spearman_correlation'].abs())
    df_selection = df_selection.sort_value(by=['target_feature', 'tri_force'], ascending=[True, False])
    
    # Sauvegarder la liste des features retenues
    output_path = 'data/features_selectionnees_benchmark.csv'
    df_selection[['target_feature', 'source_feature', 'spearman_correlation', 'kruskal_p_value', 'auc_roc', 'auc_discriminative_strength']].to_csv(output_path, index=False, sep=';')
    
    print(f"\n[SUCCÈS] {len(df_selection)} features correspondent à nos critères d'excellence !地方")
    print(f"La liste a été sauvegardée dans : {output_path}")
    
    # Affichage d'un aperçu par variable clinique
    print("\n--- NOMBRE DE FEATURES SÉLECTIONNÉES PAR CRITÉRE ---")
    print(df_selection['target_feature'].value_counts())

if __name__ == '__main__':
    main()