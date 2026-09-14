#!/usr/bin/env python3
"""
Modélisation, Équations de Substitution et Rapport de Benchmark avec Poids Normalisés
- Charge la matrice patients et le catalogue des features sélectionnées
- Entraîne une régression automatique pour chaque signe de relecture
- Extrait les poids (a, b, c) et calcule leur contribution relative en %
- Exporte un tableau CSV exhaustif et calibré pour les présentations
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def find_file(filename):
    """Cherche le fichier dans le dossier courant ou dans 'data/'"""
    if os.path.exists(filename):
        return filename
    potential_path = os.path.join('data', filename)
    if os.path.exists(potential_path):
        return potential_path
    raise FileNotFoundError(f"[ERREUR] Impossible de trouver le fichier '{filename}'.")

def main():
    print("1. Chargement des données pour la modélisation...")
    try:
        path_patients = find_file('matrice_patients_pour_regression.csv')
        path_features = find_file('features_selectionnees_benchmark.csv')
    except FileNotFoundError as e:
        print(e)
        print("[INFO] Veuillez d'abord exécuter le script de criblage statistique.")
        return

    # Chargement de la matrice patients (Lignes = Patients)
    df_patients = pd.read_csv(path_patients, sep=';', index_col='patient_num')
    
    # Chargement du dictionnaire des performances statistiques
    df_features = pd.read_csv(path_features, sep=';')

    # Liste des variables cliniques cibles à modéliser
    variables_cliniques = df_features['target_feature'].unique()
    
    print(f"[OK] {len(variables_cliniques)} variables cliniques prêtes à être modélisées.\n")
    print("=" * 90)
    print("   CONSTRUCTION DES ÉQUATIONS DE SUBSTITUTION (IMITATION DU RADIOLOGUE)")
    print("=" * 90)

    # Liste pour accumuler les résultats du tableau de synthèse CSV
    rows_benchmark = []

    for cible in variables_cliniques:
        # 1. Sélectionner le Top 3 des features radiomiques pour cette cible
        df_cible_features = df_features[df_features['target_feature'] == cible]
        # On trie par p-value croissante (les plus significatives d'abord)
        df_cible_features = df_cible_features.sort_values(by='kruskal_p_value').head(3)
        
        features_x = df_cible_features['source_feature'].tolist()
        
        if len(features_x) == 0:
            continue

        # 2. Préparation des données (X et Y) pour les patients
        df_model = df_patients[[cible] + features_x].dropna()
        
        y = df_model[cible]
        X = df_model[features_x]

        if len(y) < 10:  # Sécurité si pas assez de patients
            print(f"\n[⚠️] Pas assez de données pour modéliser {cible}.")
            continue

        # 3. Standardisation des variables radiomiques (X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. Choix du modèle selon la nature de la variable clinique
        is_binary = (y.nunique() == 2 and set(y.unique()).issubset({0, 1}))
        
        if is_binary:
            model = LogisticRegression(fit_intercept=True, penalty=None)
            model.fit(X_scaled, y)
            coefficents = model.coef_[0]
            intercept = model.intercept_[0]
            type_modele = "Logistique (Binaire)"
        else:
            model = LinearRegression(fit_intercept=True)
            model.fit(X_scaled, y)
            coefficents = model.coef_
            intercept = model.intercept_
            type_modele = "Linéaire (Continue)"

        # 5. Affichage dans la console (Équations brutes pour validation)
        print(f"\n🔹 CRITÈRE CLINIQUE : {cible}")
        print(f"   | Type de modèle : {type_modele}")
        print(f"   | Nb Patients    : {len(y)}")
        
        equation_str = f"     {cible} * (Z-score) = "
        terms = []
        for i, feat in enumerate(features_x):
            signe = "+" if coefficents[i] >= 0 else "-"
            terms.append(f"{signe} {abs(coefficents[i]):.3f} * [{feat}]")
        
        equation_str += " ".join(terms)
        signe_inter = "+" if intercept >= 0 else "-"
        equation_str += f" {signe_inter} {abs(intercept):.3f}"
        print(equation_str)
        print("-" * 90)

        # 6. Normalisation des poids (Calcul des contributions relatives absolues à 100%)
        somme_poids_abs = sum([abs(w) for w in coefficents])
        if somme_poids_abs == 0: 
            somme_poids_abs = 1  # Sécurité division par zéro

        # 7. Préparation de la ligne enrichie pour le tableau CSV de synthèse
        row_data = {
            'Variable_Clinique': cible,
            'Type_Modele': type_modele,
            'Nombre_Patients': len(y),
            'Intercept_Constante': round(intercept, 4),
            
            'Feature_1': features_x[0] if len(features_x) > 0 else None,
            'Poids_F1': round((abs(coefficents[0]) / somme_poids_abs) * 100, 1)/100 if len(features_x) > 0 else None,
            
            'Feature_2': features_x[1] if len(features_x) > 1 else None,
            'Poids_F2': round((abs(coefficents[1]) / somme_poids_abs) * 100, 1)/100 if len(features_x) > 1 else None,
            
            'Feature_3': features_x[2] if len(features_x) > 2 else None,
            'Poids_F3': round((abs(coefficents[2]) / somme_poids_abs) * 100, 1)/100 if len(features_x) > 2 else None,
        }
        rows_benchmark.append(row_data)

    # 8. Création et sauvegarde du tableau CSV récapitulatif
    df_benchmark_final = pd.DataFrame(rows_benchmark)
    os.makedirs('data', exist_ok=True)
    output_path = 'data/poids_correlation_normalises.csv'
    df_benchmark_final.to_csv(output_path, sep=';', index=False)
    
    print("\n" + "=" * 90)
    print(f"[OK] Nouveau tableau CSV avec poids normalisés enregistré : {output_path}")
    print("=" * 90)

if __name__ == '__main__':
    main()