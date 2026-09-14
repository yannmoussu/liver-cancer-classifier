import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Importation des trois nouveaux modèles
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================

df = pd.read_csv("data/flattened_global_scaled.csv", sep=";")

# Suppression des lignes de la classe "Mixtes" pour repasser au problème binaire
df = df[df["classe_name"] != "Mixtes"]

# Encodage textuel de la colonne cible ("CCK" et "CHC" -> 0 et 1)
label_encoder = LabelEncoder()
df["target"] = label_encoder.fit_transform(df["classe_name"])

# Sélection de toutes les features (en excluant les identifiants et la cible)
metadata_cols = ["id", "patient_num", "classe_name", "target"]
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols].values
y = df["target"].values

# Découpage 80% train / 20% test avec stratification binaire
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# 2. CONFIGURATION DES MODÈLES À COMPARER
# ==========================================

modeles = {
    "Support Vector Machine (SVM)": SVC(
        kernel="linear",           # Noyau non linéaire (RBF), tu peux tester 'linear' ou 'poly'
        C=0.01,                  # Paramètre de régularisation (plus C est grand, moins on tolère d'erreurs en train)
        probability=True,       # Utile si tu veux calculer des probabilités ou l'AUC-ROC plus tard
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,       # Nombre maximal d'arbres
        max_depth=2,            # Profondeur max des arbres (faible pour limiter le surapprentissage)
        learning_rate=0.01,      # Pas d'apprentissage
        eval_metric="logloss",  # Fonction de perte binaire
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1            # Évite d'encombrer la console avec les logs internes de LightGBM
    )
}

# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT ET DE PERFORMANCE
# ==========================================

for nom, modele in modeles.items():
    print("\n" + "="*60)
    print(f"ÉVALUATION DU MODÈLE : {nom}")
    print("="*60)
    
    # Entraînement du modèle
    modele.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = modele.predict(X_train)
    y_pred_test = modele.predict(X_test)
    
    # Calcul de l'Exactitude (Accuracy)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"Accuracy Train : {acc_train * 100:.2f}%")
    print(f"Accuracy Test  : {acc_test * 100:.2f}%")
    
    # Analyse de l'importance des variables (uniquement pour XGBoost et LightGBM)
    if hasattr(modele, 'feature_importances_'):
        print("\nTop 5 des caractéristiques radiomiques les plus discriminantes :")
        df_importance = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": modele.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        print(df_importance.head(5).to_string(index=False))
        
    print("\nRapport de classification détaillé (Données Test) :")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))