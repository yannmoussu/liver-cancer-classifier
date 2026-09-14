import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.validation import evaluer_modele_kfold
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from group_lasso import LogisticGroupLasso

# Wrapper pour que LogisticGroupLasso accepte la dimension y de sklearn
class SklearnGroupLasso(LogisticGroupLasso):
    def fit(self, X, y):
        # group-lasso exige un y de forme (n_samples, 1)
        return super().fit(X, np.array(y).reshape(-1, 1))

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================

df = pd.read_csv("./data/multislice_aggregated_flattened.csv", sep=";")

# Suppression des lignes de la classe "Mixtes" pour passer à un problème binaire
df = df[df["classe_name"] != "Mixtes"].copy()

# Encodage textuel de la colonne cible ("CCK" et "CHC" -> 0 et 1)
label_encoder = LabelEncoder()
df["target"] = label_encoder.fit_transform(df["classe_name"])

# Sélection de toutes les features
metadata_cols = ["id", "patient_num", "classe_name", "target"]
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols]
y = df["target"].values

# ==========================================
# 2. DÉFINITION DES GROUPES
# ==========================================
groups = []
group_dict = {}
current_group_id = 0

for col in feature_cols:
    if col in ["Age_at_disease", "Gender"]:
        group_name = col
    else:
        group_name = re.sub(r'_(mean|std|min|q25|q50|q75|max)', '', col)
        
    if group_name not in group_dict:
        group_dict[group_name] = current_group_id
        current_group_id += 1
    groups.append(group_dict[group_name])

groups = np.array(groups)

# ==========================================
# 3. LANCEMENT DE LA VALIDATION K-FOLD
# ==========================================

# Paramétrage du modèle
model = SklearnGroupLasso(
    groups=groups,
    group_reg=0.15,
    l1_reg=0.0,
    scale_reg="inverse_group_size",
    supress_warning=True,
    n_iter=1000,
    tol=1e-3,
    random_state=42
)

# Utilisation directe du fonctionnement de validation.py
evaluer_modele_kfold(
    model, 
    X, 
    y, 
    df, 
    col_group='id', 
    noms_classes=label_encoder.classes_, 
    col_age='Age_at_disease', 
    col_sexe='Gender'
)
