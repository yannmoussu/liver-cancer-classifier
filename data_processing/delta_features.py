import pandas as pd
from typing import List

def compute_phase_deltas(
    df: pd.DataFrame, 
    features: List[str], 
    phase_1: str = "PORT", 
    phase_2: str = "ART", 
    phase_3: str = "TARD"
) -> pd.DataFrame:
    """
    Calcule les deltas d'évolution temporelle pour une liste de variables
    et purge les phases intermédiaires fortement corrélées.
    """
    # Bonne pratique : Travailler sur une copie pour éviter la mutation in-place (SettingWithCopyWarning)
    df_transformed = df.copy()
    cols_to_drop = []

    for feature in features:
        col_p1 = f"{feature}_{phase_1}"
        col_p2 = f"{feature}_{phase_2}"
        col_p3 = f"{feature}_{phase_3}"
        
        # Vérification de l'existence (ex: StandardDeviation est absente)
        if all(c in df_transformed.columns for c in [col_p1, col_p2, col_p3]):
            
            # Opérations vectorisées
            df_transformed[f"{feature}_delta_{phase_2}_{phase_1}"] = df_transformed[col_p2] - df_transformed[col_p1]
            df_transformed[f"{feature}_delta_{phase_3}_{phase_2}"] = df_transformed[col_p3] - df_transformed[col_p2]
            
            # On ajoute les phases 1 et 2 à la liste des purges pour éviter la multicolinéarité
            cols_to_drop.extend([col_p1, col_p2])
        else:
            print(f"Warning : Les colonnes pour la variable '{feature}' sont incomplètes et ont été ignorées.")

    # Suppression en masse (plus performant que drop itératif)
    df_transformed.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    
    return df_transformed


if __name__ == "__main__":
    # 1. Définition des variables cibles
    delta_features = [
        "original_firstorder_Mean",
        "original_firstorder_Median",
        "original_firstorder_RootMeanSquared",
        "original_firstorder_10Percentile",
        "original_firstorder_90Percentile",
        "original_firstorder_Variance",
        "original_firstorder_StandardDeviation", # Sera ignoré automatiquement
        "original_firstorder_InterquartileRange",
        "original_firstorder_MeanAbsoluteDeviation",
        "original_firstorder_Skewness",
        "original_firstorder_Kurtosis"
    ]

    # 2. Chargement du DataFrame (séparateur point-virgule)
    df_initial = pd.read_csv("data/global_excel_resampled_normalized_flattened.csv", sep=";")
    
    # 3. Transformation
    df_final = compute_phase_deltas(df_initial, delta_features)
    
    # 4. Export
    output_filename = "data/global_excel_resampled_normalized_flattened_deltas.csv"
    df_final.to_csv(output_filename, sep=";", index=False)
    print(f"Traitement terminé. Fichier exporté : {output_filename}")