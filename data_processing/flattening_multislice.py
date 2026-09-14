import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Chargement des données
df_multislice = pd.read_csv('data/multislice_excel_basic(Sheet1).csv', sep=';')

# 2. Remplacement de VEIN par TARD
df_multislice['temps_inj'] = df_multislice['temps_inj'].replace('VEIN', 'TARD')

# 3. Filtrage : on ne conserve que les patients présents pour 3 temps_inj différents
phases_par_patient = df_multislice.groupby('id')['temps_inj'].nunique()
patients_valides = phases_par_patient[phases_par_patient == 3].index
df_multislice = df_multislice[df_multislice['id'].isin(patients_valides)]

print(f"Nombre de patients conservés avec les 3 phases : {len(patients_valides)}")

# 4. Filtrage des colonnes "diagnostic"
df = df_multislice.loc[:, ~df_multislice.columns.str.contains('diagnostic', case=False, na=False)]

# 5. Définition des fonctions pour les quantiles
def q25(x): return x.quantile(0.25)
def q50(x): return x.quantile(0.50)
def q75(x): return x.quantile(0.75)

# 6. Identification des colonnes numériques à agréger (les features radiomiques)
feature_cols = [col for col in df.columns if col.startswith('original_')]

df = df.copy()
# Conversion explicite en numérique, en remplaçant les virgules par des points si nécessaire
for col in feature_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# 7. Dictionnaire d'agrégation
aggregations = {col: ['mean', 'std', 'min', q25, q50, q75, 'max'] for col in feature_cols}

print("Agrégation en cours (calcul des moyennes, std et quantiles)...")
# Groupby sur l'individu et la phase temporelle
df_agg = df.groupby(['id', 'temps_inj']).agg(aggregations)

# Aplatissement des noms de colonnes (ex: original_shape2D_Elongation_mean)
df_agg.columns = [f"{col}_{stat.__name__ if callable(stat) else stat}" for col, stat in df_agg.columns]
df_agg = df_agg.reset_index()

# Ajout des métadonnées qui sont constantes par id/temps_inj
metadata = df[['id', 'temps_inj', 'classe_name', 'patient_num']].drop_duplicates()
df_final = pd.merge(metadata, df_agg, on=['id', 'temps_inj'], how='right')

print(f"Dimensions après agrégation (par id et phase) : {df_final.shape}")

# 6. Normalisation globale (sur les 3 phases combinées)
# Comme le DataFrame est en format "long", le StandardScaler calcule 
# la moyenne et l'écart-type sur l'ensemble des phases pour une même feature.
print("Normalisation des features...")
cols_to_normalize = [col for col in df_final.columns if col not in ['id', 'temps_inj', 'classe_name', 'patient_num']]
scaler = StandardScaler()
df_final[cols_to_normalize] = scaler.fit_transform(df_final[cols_to_normalize])

# 7. Pivotement pour que 'id' devienne la seule clé primaire
print("Pivotement des phases temporelles en colonnes...")
df_wide = df_final.pivot(index=['id', 'patient_num', 'classe_name'], columns='temps_inj')

# Reconstruction des noms de colonnes (ex: original_shape2D_Elongation_mean_PORT)
df_wide.columns = [f"{feat}_{phase}" for phase, feat in df_wide.columns]
df_wide = df_wide.reset_index()

print(f"Dimensions du dataframe final (id = clé primaire) : {df_wide.shape}")

# 8. Ajout des données cliniques (Age_at_disease, Gender)
print("Ajout des données cliniques...")
df_clinique = pd.read_csv("data/Descriptif_patients(Sheet1).csv", sep=";")

# Cast des ID en int pour assurer la fusion
df_clinique["id"] = pd.to_numeric(df_clinique["id"], errors='coerce').fillna(-1).astype(int)
df_wide["id"] = df_wide["id"].astype(int)

df_clinique_sub = df_clinique[["id", "Gender", "Age_at_disease"]]
df_wide = pd.merge(df_wide, df_clinique_sub, on="id", how="left")

# Encodage du Genre (Femme = 0, Homme = 1) en gérant les variations de casse
mapping_genre = {
    "femme": 0, "f": 0, "female": 0,
    "homme": 1, "h": 1, "male": 1, "m": 1
}
df_wide["Gender"] = df_wide["Gender"].astype(str).str.strip().str.lower().map(mapping_genre)
df_wide["Gender"] = pd.to_numeric(df_wide["Gender"], errors='coerce')

# Réorganisation pour placer les métadonnées au début du dataframe
metadata_ordonnee = ["id", "patient_num", "Gender", "Age_at_disease", "classe_name"]
cols_features = [col for col in df_wide.columns if col not in metadata_ordonnee]
df_wide = df_wide[metadata_ordonnee + cols_features]

# 9. Sauvegarde
output_path = 'data/multislice_aggregated_flattened.csv'
df_wide.to_csv(output_path, sep=';', index=False)
print(f"Pipeline de prétraitement terminée. Dataframe sauvegardé sous : {output_path}")
