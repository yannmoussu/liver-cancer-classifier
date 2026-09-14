import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from group_lasso import LogisticGroupLasso

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================

df = pd.read_csv("./data/multislice_aggregated_flattened.csv", sep=";")

# Suppression des lignes de la classe "Mixtes" pour passer à un problème binaire
df = df[df["classe_name"] != "Mixtes"].copy()

# Encodage textuel de la colonne cible ("CCK" et "CHC" -> 0 et 1)
label_encoder = LabelEncoder()
df["target"] = label_encoder.fit_transform(df["classe_name"])

# Sélection de toutes les features (en excluant les identifiants et la cible)
metadata_cols = ["id", "patient_num", "classe_name", "target"]
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols].values
y = df["target"].values.reshape(-1, 1)

# ==========================================
# 2. DÉFINITION DES GROUPES
# ==========================================
# Pour chaque feature, on retire la partie stat (_mean, _std, etc.) pour trouver son groupe
groups = []
group_dict = {}
current_group_id = 0

for col in feature_cols:
    if col in ["Age_at_disease", "Gender"]:
        group_name = col
    else:
        # Ex: original_shape2D_Elongation_mean_PORT -> original_shape2D_Elongation_PORT
        group_name = re.sub(r'_(mean|std|min|q25|q50|q75|max)', '', col)
        
    if group_name not in group_dict:
        group_dict[group_name] = current_group_id
        current_group_id += 1
    groups.append(group_dict[group_name])

groups = np.array(groups)
n_groups = len(np.unique(groups))
print(f"Nombre de groupes identifiés : {n_groups} pour {len(feature_cols)} features au total.")

# Découpage 80% train / 20% test avec stratification binaire
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 3. ENTRAÎNEMENT DU GROUP LASSO (BINAIRE)
# ==========================================

print("Entraînement du modèle LogisticGroupLasso...")
# Paramètres du Group Lasso :
# group_reg (lambda) contrôle la pénalité sur les groupes
# l1_reg contrôle la pénalité L1 standard à l'intérieur des groupes (Sparse Group Lasso)
model = LogisticGroupLasso(
    groups=groups,
    group_reg=0.15, 
    l1_reg=0.0,      # Strictement Group Lasso (pas Sparse Group Lasso)
    scale_reg="inverse_group_size",
    supress_warning=True,
    n_iter=1000,
    tol=1e-3,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 4. ÉVALUATION ET ANALYSE DES COEFFICIENTS
# ==========================================

y_pred_train = model.predict(X_train).flatten()
y_pred_test = model.predict(X_test).flatten()

acc_train = accuracy_score(y_train.flatten(), y_pred_train)
acc_test = accuracy_score(y_test.flatten(), y_pred_test)

print(f"\nAccuracy Train : {acc_train * 100:.2f}%")
print(f"Accuracy Test  : {acc_test * 100:.2f}%")

# Analyse des coefficients (shape = (n_features, n_classes))
# Pour du binaire, la classe 1 est généralement model.coef_[:, 1]
coefs = model.coef_[:, 1] if model.coef_.shape[1] > 1 else model.coef_[:, 0]

# Création d'un DataFrame pour l'analyse
df_coefs = pd.DataFrame({
    'Feature': feature_cols,
    'Group_ID': groups,
    'Coefficient': coefs
})

# Un groupe est actif si au moins une de ses features a un coef != 0
groupes_actifs = df_coefs.groupby('Group_ID')['Coefficient'].apply(lambda x: (x != 0).any()).sum()

print(f"\nSélection Group Lasso : {groupes_actifs} groupes actifs sur un total de {n_groups}")

# Filtrage pour n'afficher que les groupes actifs
df_actifs = df_coefs[df_coefs['Coefficient'] != 0].copy()

# On rajoute le nom du groupe
inverse_group_dict = {v: k for k, v in group_dict.items()}
df_actifs['Group_Name'] = df_actifs['Group_ID'].map(inverse_group_dict)

print("\n--- Features Actives (regroupées) ---")
if not df_actifs.empty:
    for group_name, group_data in df_actifs.groupby('Group_Name'):
        print(f"\nGroupe: {group_name}")
        for _, row in group_data.iterrows():
            print(f"   {row['Feature']}: {row['Coefficient']:.6f}")
else:
    print("Aucune feature sélectionnée. Essayez de réduire group_reg.")

print("\nRapport de classification détaillé (Test) :")
print(classification_report(y_test.flatten(), y_pred_test, target_names=label_encoder.classes_))
