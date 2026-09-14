#!/usr/bin/env python3
"""
Méta-classifieur CCK vs CHC à partir de pseudo-features "radiologue virtuel".

Reconstruit des pseudo-features linéaires (intercept + 3 features pondérées)
à partir d'un catalogue de poids par variable clinique, puis entraîne un
classifieur CCK (0) vs CHC (1) dessus.

--weights {optimises,correlation} : source des poids des pseudo-features.
    * optimises   : data/poids_modeles_radiomiques.csv (poids optimisés par
      optipoids.py)
    * correlation : data/parametres_correlation.csv (poids de corrélation
      bruts, issus de correlation.py)
--mode {seuls,lasso} : ce qui entre en compétition avec les pseudo-features.
    * seuls : les pseudo-features seules, régression logistique simple
      (+ courbe ROC).
    * lasso : les pseudo-features concourent avec le reste du catalogue
      radiomique brut, départagées par une régression logistique LASSO (L1)
      qui élimine les variables inutiles ("compétition Match-Play").

Exemples :
    python modeling/poids_pseudo_radiologue.py
    python modeling/poids_pseudo_radiologue.py --weights correlation --mode seuls
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.validation import evaluer_modele_kfold

warnings.filterwarnings('ignore')

# --- Les deux axes de configuration -----------------------------------------

WEIGHTS_FILES = {
    'optimises': 'poids_modeles_radiomiques.csv',
    'correlation': 'parametres_correlation.csv',
}

# Dossiers de sortie plots/ : identiques aux 4 scripts d'origine pour préserver
# l'historique des résultats déjà commités.
OUTPUT_PLOT_DIRS = {
    ('optimises', 'lasso'): 'plots/optimisés_lasso',
    ('optimises', 'seuls'): 'plots/optimisés_seuls',
    ('correlation', 'lasso'): 'plots/pseudoradiologue_lasso',
    ('correlation', 'seuls'): 'plots/pseudoradiologue_poidsfixes',
}

# C du LASSO : identique aux valeurs choisies dans les scripts d'origine.
LASSO_C = {
    'optimises': 0.5,
    'correlation': 0.4,
}


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


def load_specific_files(weights_file):
    """Charge les fichiers nécessaires depuis le dossier 'data/'."""
    base_dir = "data"

    path_global = os.path.join(base_dir, "global_excel_resampled_normalized_flattened.csv")
    path_parametres = os.path.join(base_dir, weights_file)
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


def build_pseudo_features(df_params, df_radio_scaled, patients_communs, keep_used_features):
    """Reconstruit les pseudo-features linéaires à partir du catalogue de poids.

    Si keep_used_features est True, renvoie aussi l'ensemble des features
    radiomiques consommées par les pseudo-features (mode 'lasso', pour les
    exclure du reste du catalogue brut).
    """
    print("\nCalcul des pseudo-features linéaires à partir des poids de corrélation...")
    df_pseudo_features = pd.DataFrame(index=patients_communs)
    used_features = set()

    for _, row in df_params.iterrows():
        cible = row['Variable_Clinique']
        intercept = float(row['Intercept_Constante'])

        f1, w1 = row['Feature_1'], float(row['Poids_Feature_1'])
        f2, w2 = row['Feature_2'], float(row['Poids_Feature_2'])
        f3, w3 = row['Feature_3'], float(row['Poids_Feature_3'])

        if f1 in df_radio_scaled.columns and f2 in df_radio_scaled.columns and f3 in df_radio_scaled.columns:
            z = intercept + (w1 * df_radio_scaled[f1]) + (w2 * df_radio_scaled[f2]) + (w3 * df_radio_scaled[f3])
            df_pseudo_features[f"pseudo_{cible}"] = z
            if keep_used_features:
                used_features.update([f1, f2, f3])

    if df_pseudo_features.empty:
        raise ValueError("Impossible de construire les variables composites.")

    print(f"[OK] {len(df_pseudo_features.columns)} pseudo-features radiologues générées avec succès.")
    return df_pseudo_features, used_features


def plot_pca_suite(X_scaled, y_final, output_plot_dir, title_suffix):
    """Les 3 graphiques ACP communs aux 4 configurations."""
    print("\nCalcul et génération des figures de l'ACP...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'type': y_final.map({0: 'CCk', 1: 'CHC'})
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='type', palette='Set1', s=60, alpha=0.7)
    plt.title(f'PCA of {title_suffix} (CCk vs CHC)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(f'{output_plot_dir}/pca_chc_cck_pseudo_features.png', dpi=300)
    plt.close()

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

    pca_full = PCA().fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    components = range(1, min(len(explained_var) + 1, 30))
    plt.bar(components, explained_var[:29], alpha=0.7, color='steelblue')
    plt.step(components, np.cumsum(explained_var[:29]), where='mid', label='Cumulative variance', color='orange', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'PCA Explained Variance (Scree Plot) - {title_suffix}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_plot_dir}/pca_explained_variance.png', dpi=300)
    plt.close()

    print(f"[OK] Les 3 graphiques ACP ont été enregistrés dans {output_plot_dir}/")


def run_lasso_mode(df_params, df_radio_scaled, df_target, patients_communs, output_plot_dir, weights):
    """Pseudo-features en compétition avec le reste du catalogue radiomique brut (LASSO)."""
    df_pseudo_features, used_features = build_pseudo_features(
        df_params, df_radio_scaled, patients_communs, keep_used_features=True
    )

    exclude_metadata = ['id', 'patient_num', 'id_x', 'id_y', 'Age_at_disease', 'Gender']
    raw_features_cols = [c for c in df_radio_scaled.columns if c not in used_features and c not in exclude_metadata]
    df_raw_features = df_radio_scaled[raw_features_cols]

    X_competition = pd.concat([df_pseudo_features, df_raw_features], axis=1)
    y_final = df_target['target'].astype(int)

    print(f"Taille de la matrice de compétition : {X_competition.shape} "
          f"({len(df_pseudo_features.columns)} pseudo-features, {len(raw_features_cols)} features brutes)")

    scaler_final = StandardScaler()
    X_competition_scaled = pd.DataFrame(
        scaler_final.fit_transform(X_competition),
        index=X_competition.index,
        columns=X_competition.columns
    )

    c_value = LASSO_C[weights]
    print(f"\nEntraînement de la régression logistique pénalisée LASSO (C={c_value})...")
    lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', C=c_value, random_state=42, max_iter=2000)
    lasso_clf.fit(X_competition_scaled, y_final)

    y_pred = lasso_clf.predict(X_competition_scaled)

    print("\n" + "=" * 80)
    print("   RAPPORT DE PERFORMANCE DE LA COMPÉTITION (MATCH-PLAY LASSO)")
    print("=" * 80)
    print(classification_report(y_final, y_pred, target_names=['CCk', 'CHC']))
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_final, y_pred))

    coefs = lasso_clf.coef_[0]
    df_coefficients = pd.DataFrame({
        'Feature': X_competition.columns,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    })
    df_survivants = df_coefficients[df_coefficients['Coefficient'] != 0].sort_values(by='Abs_Coefficient', ascending=False)
    print(f"\n[LASSO] Nombre de variables sélectionnées (survivantes) : {len(df_survivants)} / {X_competition.shape[1]}")
    print(df_survivants[['Feature', 'Coefficient']])

    # Nom de fichier distinct par source de poids (les 2 scripts d'origine
    # s'écrasaient l'un l'autre en écrivant tous les deux dans
    # data/survivants_lasso_competition.csv).
    df_survivants_csv = df_survivants.copy()
    df_survivants_csv['Coefficient'] = df_survivants_csv['Coefficient'].astype(str).str.replace('.', ',')
    df_survivants_csv['Abs_Coefficient'] = df_survivants_csv['Abs_Coefficient'].astype(str).str.replace('.', ',')
    survivants_path = f"data/survivants_lasso_competition_{weights}.csv"
    df_survivants_csv.to_csv(survivants_path, sep=';', index=False)
    print(f"[OK] Coefficients survivants exportés dans {survivants_path}")

    if not df_survivants.empty:
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        df_plot = df_survivants.head(20).copy()
        df_plot['Color_Group'] = df_plot['Coefficient'].apply(lambda x: '#d62728' if x >= 0 else '#1f77b4')
        sns.barplot(
            data=df_plot,
            x='Coefficient',
            y='Feature',
            palette=df_plot['Color_Group'].tolist(),
            hue='Feature',
            legend=False
        )
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.2)
        plt.title("Top 20 des variables sélectionnées par le LASSO\n(Rouge = Oriente CHC | Bleu = Oriente CCk)",
                   fontsize=13, fontweight='bold')
        plt.xlabel("Coefficient dans le modèle")
        plt.ylabel("Nom de la Feature")
        plt.tight_layout()
        plt.savefig(f'{output_plot_dir}/competition_lasso_survivants.png', dpi=300)
        plt.close()
        print(f"[OK] Graphique de la compétition enregistré dans {output_plot_dir}/")

    plot_pca_suite(X_competition_scaled, y_final, output_plot_dir, "Competition Space")

    return lasso_clf, X_competition_scaled, y_final


def run_seuls_mode(df_params, df_radio_scaled, df_target, patients_communs, output_plot_dir, weights):
    """Les pseudo-features seules, méta-classifieur linéaire simple."""
    df_pseudo_features, _ = build_pseudo_features(
        df_params, df_radio_scaled, patients_communs, keep_used_features=False
    )

    X_final = df_pseudo_features
    y_final = df_target['target'].astype(int)

    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    meta_clf = LogisticRegression(random_state=42, max_iter=1000)
    meta_clf.fit(X_final_scaled, y_final)

    y_pred = meta_clf.predict(X_final_scaled)
    y_probs = meta_clf.predict_proba(X_final_scaled)[:, 1]

    print("\n" + "=" * 80)
    print("   RAPPORT DE PERFORMANCE DU CLASSIFIEUR COMPOSITE (PSEUDO-RADIOLOGUE)")
    print("=" * 80)
    print(classification_report(y_final, y_pred, target_names=['CCk', 'CHC']))
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_final, y_pred))

    coefs = meta_clf.coef_[0]
    df_poids_finaux = pd.DataFrame({
        'Critere_Semiologique': [c.replace('pseudo_', '') for c in X_final.columns],
        'Coefficient_Brut': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values(by='Abs_Coefficient', ascending=False)

    # Nom de fichier distinct par source de poids, même raison que côté LASSO.
    df_poids_finaux_csv = df_poids_finaux.copy()
    df_poids_finaux_csv['Coefficient_Brut'] = df_poids_finaux_csv['Coefficient_Brut'].astype(str).str.replace('.', ',')
    df_poids_finaux_csv['Abs_Coefficient'] = df_poids_finaux_csv['Abs_Coefficient'].astype(str).str.replace('.', ',')
    poids_path = f"data/poids_decisionnels_chc_cck_lineaire_{weights}.csv"
    df_poids_finaux_csv.to_csv(poids_path, sep=';', index=False)
    print(f"[OK] Poids décisionnels exportés dans {poids_path}")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['#d62728' if c >= 0 else '#1f77b4' for c in df_poids_finaux['Coefficient_Brut']]
    sns.barplot(data=df_poids_finaux.head(15), x='Coefficient_Brut', y='Critere_Semiologique', ax=ax1,
                palette=colors, hue='Critere_Semiologique', legend=False)
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

    X_final_scaled_df = pd.DataFrame(X_final_scaled, columns=X_final.columns, index=X_final.index)
    plot_pca_suite(X_final_scaled_df, y_final, output_plot_dir, "Composite Features")

    return meta_clf, X_final_scaled_df, y_final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', choices=['optimises', 'correlation'], default='optimises',
                         help="Source des poids pour les pseudo-features (défaut : optimises)")
    parser.add_argument('--mode', choices=['seuls', 'lasso'], default='lasso',
                         help="Pseudo-features seules, ou en compétition LASSO avec le catalogue brut (défaut : lasso)")
    args = parser.parse_args()

    try:
        df_glob, df_params, df_relec = load_specific_files(WEIGHTS_FILES[args.weights])
    except FileNotFoundError as e:
        print(e)
        return

    output_plot_dir = OUTPUT_PLOT_DIRS[(args.weights, args.mode)]
    os.makedirs(output_plot_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. PREPROCESSING DES RELECTURES
    # -------------------------------------------------------------------------
    df_relec['classe_name'] = df_relec['classe_name'].astype(str).str.strip()
    df_relec = df_relec[df_relec['classe_name'].isin(['CHC', 'CCK'])].copy()
    df_relec = df_relec.replace('', np.nan).fillna(0)
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

    scaler_radio = StandardScaler()
    df_radio_scaled = pd.DataFrame(
        scaler_radio.fit_transform(df_radio),
        index=df_radio.index,
        columns=df_radio.columns
    )

    # -------------------------------------------------------------------------
    # 4-8. Spécifique au mode choisi
    # -------------------------------------------------------------------------
    if args.mode == 'lasso':
        model, X_scaled, y_final = run_lasso_mode(
            df_params, df_radio_scaled, df_target, patients_communs, output_plot_dir, args.weights
        )
    else:
        model, X_scaled, y_final = run_seuls_mode(
            df_params, df_radio_scaled, df_target, patients_communs, output_plot_dir, args.weights
        )

    # -------------------------------------------------------------------------
    # 9. VALIDATION CROISÉE (Stratified Group K-Fold)
    # -------------------------------------------------------------------------
    print("\n=== Évaluation par Cross-Validation (Stratified Group K-Fold) ===")
    evaluer_modele_kfold(
        modele=model,
        X=X_scaled,
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
