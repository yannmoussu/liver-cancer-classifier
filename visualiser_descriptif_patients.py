import pandas as pd

df_norm = pd.read_csv("data/global_excel_resampled(Sheet1).csv", sep = ";")

col_a_garder = [col for col in df_norm.columns if col.startswith("original")]

col_first = [col for col in df_norm.columns if "firstorder" in col and "entropy" not in col.lower()]

df_norm = df_norm[col_a_garder + ['classe_name', 'temps_inj', 'patient_num']]

for col in df_norm.columns:
    if df_norm[col].dtype == 'object':
        # 1. On remplace les virgules par des points
        col_nettoyee = df_norm[col].astype(str).str.replace(',', '.')
        
        # 2. On force la conversion en nombre. 
        # Les textes comme 'Mixtes' deviendront 'NaN' (vides) sans faire planter le code.
        df_norm[col] = pd.to_numeric(col_nettoyee, errors='ignore')


valeurs_groupees = df_norm[col_first].values.flatten()
serie_globale = pd.Series(valeurs_groupees)

moyenne_globale = serie_globale.mean()
ecart_type_global = serie_globale.std()

for col in df_norm.columns:
    if 'firstorder' in col and 'entropy' not in col.lower():
        df_norm[col] = (df_norm[col] - moyenne_globale) / ecart_type_global

    elif col in ['classe_name', 'temps_inj', 'patient_num'] :
        continue
    else :
        mean = df_norm[col].mean()
        std = df_norm[col].std()

        if std != 0:
            df_norm[col] = (df_norm[col] - mean) / std
        
        else :
            df_norm[col] = 0.0

df_norm.to_csv("data/global_excel_resampled_normalized.csv", sep = ";", index = False)
