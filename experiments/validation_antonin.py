import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from common.validation import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================

df = pd.read_csv("./data/flatted_global_scaled_uncor_08.csv", sep=";")

# Suppression des lignes de la classe "Mixtes" pour passer à un problème binaire
df = df[df["classe_name"] != "Mixtes"]

# Encodage textuel de la colonne cible ("CCK" et "CHC" -> 0 et 1)
label_encoder = LabelEncoder()
df["target"] = label_encoder.fit_transform(df["classe_name"])

# Sélection de toutes les features (en excluant les identifiants et la cible)
metadata_cols = ["id", "patient_num", "classe_name", "target"]
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols]
y = df["target"]

model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.9,
    random_state=42,
)

evaluer_modele_kfold(model, X, y, df, col_group='id', noms_classes=['CHC', 'CCK'], col_age='Age_at_disease', col_sexe='Gender')