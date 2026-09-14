import pandas as pd
import matplotlib.pyplot as plt

df_global = pd.read_csv('data/global_excel_resampled_scaled.csv', sep=';')
df_multislice = pd.read_csv('data/multislice_excel_basic(Sheet1).csv', sep=';')

global_cols = set(df_global.columns)
multi_cols = set(df_multislice.columns)

print(f"Colonnes communes : {len(global_cols.intersection(multi_cols))}")
print(f"Colonnes uniquement dans Global : {len(global_cols - multi_cols)}")
print(f"Colonnes uniquement dans Multislice : {len(multi_cols - global_cols)}")

comptage = df_multislice['id'].value_counts()
print("\nComptage par ID :")
print(comptage)

# Création de l'histogramme pour le comptage
plt.figure(figsize=(10, 6))
# On utilise bins=range pour avoir une barre par valeur entière possible (1, 2, 3...)
plt.hist(comptage, bins=50, edgecolor='black', align='left', color='skyblue')
plt.title("Distribution du nombre de lignes (slices) par patient", fontweight='bold', fontsize=14)
plt.xlabel("Nombre de lignes par patient", fontsize=12)
plt.ylabel("Nombre de patients", fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# Création du dataframe df sans les colonnes "diagnostic"
df = df_multislice.loc[:, ~df_multislice.columns.str.contains('diagnostic', case=False, na=False)]
print(f"\nDimensions du nouveau df (sans 'diagnostic') : {df.shape}")

print("\nNoms des colonnes conservées dans df :")
for col in df.columns:
    print(f" - {col}")
