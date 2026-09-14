import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# 1. Chargement et fusion des données
df_clinique = pd.read_csv(r"data\Relectures_imageries.csv", sep=";")
df_radiomique = pd.read_csv(r"data\global_excel_resampled_normalized_flattened_deltas.csv", sep=";")

df_global = pd.merge(df_clinique, df_radiomique, on="id")

# 2. Le dictionnaire complet
correspondances = {
    "APHE_heterogeneous": ["original_shape_Maximum2DDiameterRow_ART", "original_shape_SurfaceVolumeRatio_TARD", "original_glrlm_GrayLevelNonUniformity_PORT"],
    "Bile_duct_dilatation": ["original_glszm_LargeAreaLowGrayLevelEmphasis_PORT", "original_gldm_LargeDependenceLowGrayLevelEmphasis_TARD", "original_glrlm_LongRunLowGrayLevelEmphasis_TARD"],
    "Blood": ["original_firstorder_Minimum_ART", "original_ngtdm_Busyness_ART", "original_ngtdm_Busyness_PORT"],
    "Capsule": ["original_shape_Sphericity_TARD", "original_shape_Sphericity_PORT", "original_shape_Flatness_TARD"],
    "Diffusion_targetoid": ["original_glcm_InverseVariance_PORT", "original_glcm_InverseVariance_ART", "original_glcm_InverseVariance_TARD"],
    "Intratumoral_fat": ["original_glcm_ClusterShade_TARD", "original_firstorder_Skewness_delta_ART_PORT", "original_glcm_ClusterShade_PORT"],
    "LR-5": ["original_firstorder_Median_TARD", "original_firstorder_Median_delta_TARD_ART", "original_gldm_LargeDependenceLowGrayLevelEmphasis_PORT"],
    "LR-M": ["original_ngtdm_Busyness_PORT", "original_firstorder_Median_TARD", "original_glszm_LargeAreaLowGrayLevelEmphasis_PORT"],
    "Late_enhancement": ["original_gldm_LargeDependenceHighGrayLevelEmphasis_TARD", "original_glrlm_LongRunHighGrayLevelEmphasis_TARD", "original_glszm_SmallAreaLowGrayLevelEmphasis_ART"],
    "Necrosis": ["original_shape_SurfaceVolumeRatio_TARD", "original_shape_SurfaceVolumeRatio_ART", "original_shape_SurfaceVolumeRatio_PORT"],
    "Nonperiph_washout": ["original_glszm_LargeAreaLowGrayLevelEmphasis_PORT", "original_firstorder_Skewness_TARD", "original_firstorder_10Percentile_delta_TARD_ART"],
    "Portal_thrombosis": ["original_shape_MajorAxisLength_ART", "original_shape_Maximum3DDiameter_ART", "original_shape_Maximum2DDiameterSlice_ART"],
    "Rim_APHE": ["original_firstorder_Median_TARD", "original_firstorder_Mean_TARD", "original_glszm_LargeAreaLowGrayLevelEmphasis_PORT"],
    "Satellite_nodule": ["original_shape_Sphericity_ART", "original_shape_Maximum3DDiameter_ART", "original_shape_Maximum3DDiameter_PORT"],
    "Shape": ["original_shape_Sphericity_ART", "original_shape_Sphericity_PORT"],
    "Size_mm": ["original_shape_SurfaceArea_TARD", "original_shape_SurfaceArea_PORT", "original_shape_SurfaceArea_ART"],
    "nonrim_APHE": ["original_firstorder_Median_TARD", "original_gldm_DependenceVariance_PORT", "original_gldm_DependenceNonUniformityNormalized_PORT"]
}

# 3. Création de la liste pour stocker les résultats
lignes_csv = []

print("==========================================================")
print("PERFORMANCES DES MODÈLES (AFFICHAGE CONSOLE)")
print("==========================================================\n")

# 4. Boucle de modélisation
for variable_y, features_x in correspondances.items():
    data_modele = df_global[[variable_y] + features_x].dropna()
    
    if len(data_modele) < 5:
        print(f"[!] {variable_y} : Pas assez de données disponibles.")
        continue
        
    X = data_modele[features_x]
    Y = data_modele[variable_y]
    valeurs_uniques = Y.unique()
    
    ligne_actuelle = {
        "Variable_Clinique": variable_y,
        "Nombre_Patients": len(data_modele)
    }
    
    # Classification (Régression Logistique)
    if len(valeurs_uniques) <= 5:
        modele = LogisticRegression(max_iter=1000, penalty="l2")
        modele.fit(X, Y)
        predictions = modele.predict(X)
        
        # Calcul et affichage immédiat de l'Accuracy
        acc = accuracy_score(Y, predictions)
        print(f"{variable_y:<22} | Modèle: Logistique | Accuracy: {acc * 100:.2f}%")
        
        ligne_actuelle["Type_Modele"] = "Logistique"
        ligne_actuelle["Intercept_Constante"] = round(modele.intercept_[0], 4)
        poids = modele.coef_[0]
        
    # Régression Continue (Linéaire)
    else:
        modele = LinearRegression()
        modele.fit(X, Y)
        predictions = modele.predict(X)
        
        # Calcul et affichage immédiat de l'erreur des moindres carrés (MSE)
        mse = mean_squared_error(Y, predictions)
        print(f"{variable_y:<22} | Modèle: Linéaire   | Erreur Moindres Carrés (MSE): {mse:.4f}")
        
        ligne_actuelle["Type_Modele"] = "Linéaire"
        ligne_actuelle["Intercept_Constante"] = round(modele.intercept_, 4)
        poids = modele.coef_
        
    # Ajout dynamique des features et des poids pour le CSV
    for index, (nom_feature, valeur_poids) in enumerate(zip(features_x, poids)):
        numero = index + 1
        ligne_actuelle[f"Feature_{numero}"] = nom_feature
        ligne_actuelle[f"Poids_Feature_{numero}"] = round(valeur_poids, 4)
        
    lignes_csv.append(ligne_actuelle)

# 5. Conversion et sauvegarde du CSV (sans les métriques)
df_resultats = pd.DataFrame(lignes_csv)

colonnes_de_base = ["Variable_Clinique", "Type_Modele", "Nombre_Patients", "Intercept_Constante"]
colonnes_features = [col for col in df_resultats.columns if col.startswith("Feature_") or col.startswith("Poids_Feature_")]
df_resultats = df_resultats[colonnes_de_base + colonnes_features]

df_resultats = df_resultats.fillna("")
df_resultats.to_csv("data/poids_modeles_radiomiques.csv", sep=";", index=False, encoding="utf-8-sig")

