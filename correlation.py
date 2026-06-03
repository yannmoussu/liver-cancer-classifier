#!/usr/bin/env python3
"""
Analyse statistique avancée et exportation de matrices pour modélisation (Régression).
- Génère la matrice de synthèse demandée (Lignes = Clinique, Colonnes = Radiomique).
- Exporte la matrice Patient complète et alignée pour entraîner la régression.
- Conserve la génération des 4 graphiques de criblage et heatmaps épurées.
- Exporte les fichiers graphiques (.png) dans plots/correlation et les données (.csv) dans data/correlation.
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Désactiver les avertissements de conversion
warnings.filterwarnings('ignore')

# Définition des répertoires cibles demandés
DIR_PLOTS = os.path.join('plots', 'correlation')
DIR_DATA = os.path.join('data', 'correlation')

def find_file(filename):
    """Cherche le fichier à la racine ou dans un sous-dossier 'data'"""
    if os.path.exists(filename):
        return filename
    potential_path = os.path.join('data', filename)
    if os.path.exists(potential_path):
        return potential_path
    raise FileNotFoundError(f"[ERREUR] Impossible de trouver le fichier '{filename}'.")

def get_p_value_asterisks(p_val):
    """Retourne la notation par étoiles standard pour la significativité"""
    if pd.isna(p_val) or p_val >= 0.05:
        return ""
    elif p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    else:
        return "*"

def generate_plots(df_all, df_selected, liste_complete_clinique):
    """Génère les graphiques d'illustration du benchmark"""
    print("\n3. Génération des graphiques d'illustration...")
    sns.set_theme(style="whitegrid")
    
    # Sécurité : Création du dossier cible pour les plots
    os.makedirs(DIR_PLOTS, exist_ok=True)
    
    top_prioritaire = ['Nonperiph_washout', 'Late_enhancement', 'Capsule', 'LR-M']
    selected_pairs = set(df_selected['source_feature'] + "||" + df_selected['target_feature'])
    
    df_all['Statut'] = np.where(
        (df_all['source_feature'] + "||" + df_all['target_feature']).isin(selected_pairs), 
        'Sélectionné (Biomarqueur)', 'Rejeté (Bruit)'
    )
    df_all['Groupe_Clinique'] = np.where(df_all['target_feature'].isin(top_prioritaire), 'Top LogReg', 'Autres critères')
    df_all['minus_log_p'] = -np.log10(df_all['kruskal_p_value'].astype(float) + 1e-15)

    # GRAPHIC 1A : CORRÉLATION
    plt.figure(figsize=(12, 8))
    df_all['abs_spearman'] = df_all['spearman_correlation'].abs()
    sns.scatterplot(
        data=df_all, x='abs_spearman', y='minus_log_p', hue='Statut', style='Groupe_Clinique',
        markers={'Autres critères': 'o', 'Top LogReg': 'X'},
        palette={'Rejeté (Bruit)': '#b0bec5', 'Sélectionné (Biomarqueur)': '#2e7d32'}, alpha=0.75, s=80
    )
    plt.axvline(x=0.60, color='#d32f2f', linestyle='--', linewidth=1.5, label='Seuil multiclasse (|Rho| = 0.60)')
    plt.axvline(x=0.80, color='#6a1b9a', linestyle='-.', linewidth=1.5, label='Seuil taille (|Rho| = 0.80)')
    plt.axhline(y=-np.log10(0.05), color='#e65100', linestyle='--', linewidth=1.5, label='Seuil p-value (p = 0.05)')
    plt.title("Criblage du Catalogue Radiomique : Axe Corrélation", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("|Coefficient de Spearman|", fontsize=12)
    plt.ylabel("Significativité [-log10(p-value)]", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, 'visualisation_criblage_par_correlation.png'), dpi=300)
    plt.close()

    # GRAPHIC 1B : AUC
    plt.figure(figsize=(12, 8))
    df_auc_only = df_all[df_all['auc_roc'].notna()].copy()
    if not df_auc_only.empty:
        sns.scatterplot(
            data=df_auc_only, x='auc_roc', y='minus_log_p', hue='Statut', style='Groupe_Clinique',
            markers={'Autres critères': 'o', 'Top LogReg': 'X'},
            palette={'Rejeté (Bruit)': '#b0bec5', 'Sélectionné (Biomarqueur)': '#2e7d32'}, alpha=0.75, s=80
        )
        plt.axvline(x=0.65, color='#d32f2f', linestyle='--', linewidth=1.5, label="Seuil AUC = 0.65")
        plt.axvline(x=0.35, color='#d32f2f', linestyle='--', linewidth=1.5, label="Seuil AUC = 0.35")
        plt.axhline(y=-np.log10(0.05), color='#e65100', linestyle='--', linewidth=1.5, label='Seuil p-value')
        plt.title("Criblage du Catalogue Radiomique : Axe Performance Diagnostic", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("AUC-ROC", fontsize=12)
        plt.ylabel("Significativité [-log10(p-value)]", fontsize=12)
        plt.legend(loc='upper center', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_PLOTS, 'visualisation_criblage_par_auc.png'), dpi=300)
        plt.close()

    # GRAPHIC 2 : FOCUS HEATMAP
    df_top5_selected = df_selected[df_selected['target_feature'].isin(top_prioritaire)].copy()
    df_top5_best = df_top5_selected.groupby('target_feature').head(3).copy()
    if not df_top5_best.empty:
        pivot_top5 = df_top5_best.pivot_table(index='source_feature', columns='target_feature', values='auc_roc', aggfunc='first')
        pivot_top5_p = df_top5_best.pivot_table(index='source_feature', columns='target_feature', values='kruskal_p_value', aggfunc='first')
        colonnes_top5_ordonnees = [c for c in top_prioritaire if c in pivot_top5.columns]
        pivot_top5 = pivot_top5[colonnes_top5_ordonnees]
        annot_top5 = pivot_top5.copy().astype(str)
        for col in pivot_top5.columns:
            for idx in pivot_top5.index:
                val = pivot_top5.loc[idx, col]
                p = pivot_top5_p.loc[idx, col]
                annot_top5.loc[idx, col] = f"{val:.2f}{get_p_value_asterisks(p)}" if not pd.isna(val) else ""
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_top5.fillna(0), annot=annot_top5, fmt="", cmap="coolwarm", center=0.5, linewidths=0.75)
        plt.xticks(rotation=25, ha='right')
        plt.title("Focus Benchmark : Biomarqueurs Validés (Modèle LogReg)", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_PLOTS, 'heatmap_focus_top5_logreg.png'), dpi=300)
        plt.close()

    # GRAPHIC 3 : HEATMAP EXHAUSTIVE
    df_exhaustif_best = df_selected.groupby('target_feature').head(3).copy()
    if not df_exhaustif_best.empty:
        df_exhaustif_best['valeur_affichage'] = df_exhaustif_best['auc_discriminative_strength'].fillna(df_exhaustif_best['spearman_correlation'].abs())
        pivot_all = df_exhaustif_best.pivot_table(index='source_feature', columns='target_feature', values='valeur_affichage', aggfunc='first')
        pivot_all_p = df_exhaustif_best.pivot_table(index='source_feature', columns='target_feature', values='kruskal_p_value', aggfunc='first')
        ordre_affichage_final = [c for c in top_prioritaire if c in liste_complete_clinique] + [c for c in liste_complete_clinique if c not in top_prioritaire]
        pivot_all = pivot_all.reindex(columns=ordre_affichage_final)
        annot_all = pd.DataFrame("", index=pivot_all.index, columns=pivot_all.columns)
        for col in pivot_all.columns:
            for idx in pivot_all.index:
                if col in pivot_all.columns and idx in pivot_all.index:
                    val = pivot_all.loc[idx, col]
                    p = pivot_all_p.loc[idx, col] if pivot_all_p is not None else np.nan
                    if not pd.isna(val):
                        matching_rows = df_exhaustif_best[(df_exhaustif_best['source_feature']==idx) & (df_exhaustif_best['target_feature']==col)]
                        if not matching_rows.empty:
                            if col in ['Size_mm', 'Shape']:
                                annot_all.loc[idx, col] = f"{matching_rows['spearman_correlation'].values[0]:.2f}{get_p_value_asterisks(p)}"
                            else:
                                annot_all.loc[idx, col] = f"{matching_rows['auc_roc'].values[0]:.2f}{get_p_value_asterisks(p)}"
        plt.figure(figsize=(max(12, len(ordre_affichage_final)*1.3), max(9, len(pivot_all)*0.38)))
        ax = sns.heatmap(
            pivot_all.fillna(0.5),
            annot=annot_all,
            fmt="",
            cmap="coolwarm",
            center=0.5,
            linewidths=0.5,
            cbar_kws={"label": "Valeur absolue de AUC-ROC"},
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18) 

        ax.collections[0].colorbar.set_label("|AUC-ROC|", fontsize=18, fontweight="bold")
        plt.title("Cartographie Épurée du Benchmark Radiomique", fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=35, ha='right', fontsize=18)        
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_PLOTS, 'heatmap_toutes_variables_cliniques.png'), dpi=300)
        plt.close()


def main():
    try:
        flattened_path = find_file('global_excel_resampled_normalized_flattened_deltas.csv')
        relectures_path = find_file('Relectures_imageries(Feuil1).csv')
    except FileNotFoundError as e:
        print(e)
        return

    # Sécurité : Création du dossier cible pour les CSV de données
    os.makedirs(DIR_DATA, exist_ok=True)

    # 1. Chargement et nettoyage des fichiers sources
    df_radio_raw = pd.read_csv(flattened_path, sep=';', low_memory=False)
    df_radio_raw.columns = df_radio_raw.columns.str.strip()
    df_radio_raw = df_radio_raw.set_index('patient_num')
    df_radio_raw = df_radio_raw[~df_radio_raw.index.duplicated(keep='first')]

    df_clinique_raw = pd.read_csv(relectures_path, sep=';')
    df_clinique_raw.columns = df_clinique_raw.columns.str.strip()

    exclure_cles = ['Patient_number', 'id', 'patient_num', 'patient_id', 'Type_tumeur', 'T1_signal_intensity', 'T2_signal_intensity']
    cols_cliniques = [c for c in df_clinique_raw.columns if c not in exclure_cles]
    
    for col in cols_cliniques:
        df_clinique_raw[col] = pd.to_numeric(df_clinique_raw[col], errors='coerce').fillna(0)

    df_clinique_raw = df_clinique_raw[['Patient_number'] + cols_cliniques].set_index('Patient_number')
    df_clinique_raw = df_clinique_raw[~df_clinique_raw.index.duplicated(keep='first')]

    # 2. Alignement strict des patients
    common_patients = df_radio_raw.index.intersection(df_clinique_raw.index)
    df_radio = df_radio_raw.loc[common_patients]
    df_clinique = df_clinique_raw.loc[common_patients]
    radio_numeric_cols = df_radio.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in radio_numeric_cols: radio_numeric_cols.remove('id')

    print("1. Calculs statistiques et construction des matrices...")
    records = []
    
    # Création de la structure de stockage pour la matrice de synthèse demandée
    matrice_synthese = pd.DataFrame(index=cols_cliniques, columns=radio_numeric_cols)

    for src_col in radio_numeric_cols:
        x = df_radio[src_col]
        if x.nunique() <= 1: continue
            
        for tgt_col in df_clinique.columns:
            y = df_clinique[tgt_col]
            
            spearman_corr = x.corr(y, method='spearman')
            groups = [x[y == val].values for val in y.unique() if len(x[y == val]) > 0]
            kruskal_p = kruskal(*groups)[1] if len(groups) > 1 else np.nan
                
            is_binary = (y.nunique() == 2 and set(y.unique()).issubset({0, 1}))
            if is_binary:
                try:
                    auc_val = roc_auc_score(y, x.fillna(x.mean()))
                    abs_auc_effect = abs(auc_val - 0.5) + 0.5
                except Exception:
                    auc_val, abs_auc_effect = np.nan, np.nan
            else:
                auc_val, abs_auc_effect = np.nan, np.nan

            if not np.isnan(spearman_corr):
                records.append({
                    'source_feature': src_col, 'target_feature': tgt_col,
                    'spearman_correlation': float(spearman_corr),
                    'kruskal_p_value': float(kruskal_p) if not np.isnan(kruskal_p) else np.nan,
                    'auc_roc': float(auc_val) if not np.isnan(auc_val) else np.nan,
                    'auc_discriminative_strength': float(abs_auc_effect) if not np.isnan(abs_auc_effect) else np.nan
                })
                
                # Remplissage de la cellule de synthèse demandée par l'utilisateur
                p_str = f"{kruskal_p:.4f}" if not np.isnan(kruskal_p) else "NaN"
                if is_binary and not np.isnan(auc_val):
                    matrice_synthese.loc[tgt_col, src_col] = f"AUC={auc_val:.2f} | p={p_str}"
                else:
                    matrice_synthese.loc[tgt_col, src_col] = f"rs={spearman_corr:.2f} | p={p_str}"

    df_all = pd.DataFrame(records)

    # 3. Sauvegarde de la matrice de synthèse statistique demandée dans data/correlation
    path_matrice_synthese = os.path.join(DIR_DATA, 'matrice_synthese_statistiques.csv')
    matrice_synthese.to_csv(path_matrice_synthese, sep=';')
    print(f"[OK] Matrice de synthèse enregistrée : {path_matrice_synthese}")

    # 4. EXPORTATION DE LA MATRICE PATIENTS dans data/correlation
    matrice_regression_patients = pd.concat([df_clinique, df_radio[radio_numeric_cols]], axis=1)
    path_matrice_patients = os.path.join(DIR_DATA, 'matrice_patients_pour_regression.csv')
    matrice_regression_patients.to_csv(path_matrice_patients, sep=';', index_label='patient_num')
    print(f"[OK] Base Patients enregistrée pour Régression : {path_matrice_patients}")

    # 5. Filtrage d'excellence pour les graphiques et export final dans data/correlation
    filtre_taille = (df_all['target_feature'] == 'Size_mm') & (df_all['spearman_correlation'].abs() >= 0.80) & (df_all['kruskal_p_value'] < 0.05)
    filtre_binaire = (df_all['target_feature'] != 'Size_mm') & (~df_all['auc_discriminative_strength'].isna()) & (df_all['auc_discriminative_strength'] >= 0.65) & (df_all['kruskal_p_value'] < 0.05)
    filtre_multiclasse = (df_all['target_feature'] != 'Size_mm') & (df_all['auc_discriminative_strength'].isna()) & (df_all['spearman_correlation'].abs() >= 0.60) & (df_all['kruskal_p_value'] < 0.05)
    
    df_selected = df_all[filtre_taille | filtre_binaire | filtre_multiclasse].copy()
    df_selected['tri_force'] = df_selected['auc_discriminative_strength'].fillna(df_selected['spearman_correlation'].abs())
    df_selected = df_selected.sort_values(by=['target_feature', 'tri_force'], ascending=[True, False])
    
    path_features_selectionnees = os.path.join(DIR_DATA, 'features_selectionnees_benchmark.csv')
    df_selected.to_csv(path_features_selectionnees, index=False, sep=';')
    print(f"[OK] Caractéristiques sélectionnées enregistrées : {path_features_selectionnees}")
    
    # Appel de la fonction graphique
    generate_plots(df_all, df_selected, cols_cliniques)
    print("\n[SUCCÈS] Script exécuté avec succès. Vos graphiques et tables sont compartimentés dans 'plots/correlation' et 'data/correlation'.")

if __name__ == '__main__':
    main()