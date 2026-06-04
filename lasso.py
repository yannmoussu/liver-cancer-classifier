import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from validation import evaluer_modele_kfold

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================

df = pd.read_csv("./data/global_excel_resampled_normalized_flattened.csv", sep=";")

# Suppression des lignes de la classe "Mixtes" pour passer à un problème binaire
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
# 2. ENTRAÎNEMENT DE LA RÉGRESSION LOGISTIQUE LASSO (BINAIRE)
# ==========================================

# penalty='l1' active la régularisation Lasso
# solver='liblinear' est particulièrement robuste et rapide pour les problèmes binaires avec L1
# C=0.5 contrôle la force de régularisation (plus C est petit, plus la pénalité est forte)
model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.1,
    random_state=42,
)

model.fit(X_train, y_train)

# Création d'un DataFrame pour associer les colonnes à leurs coefficients
df_coefs = pd.DataFrame(
    {"Feature": feature_cols, "Coefficient": model.coef_[0]}
)

# Filtrer pour ne garder que les coefficients actifs (différents de zéro)
df_actifs = df_coefs[df_coefs["Coefficient"] != 0]

# Trier par valeur absolue pour voir les variables les plus discriminantes en premier
df_actifs = df_actifs.reindex(
    df_actifs["Coefficient"].abs().sort_values(ascending=False).index
)

# Affichage des résultats
print(f"Nombre de features actives : {len(df_actifs)}")
print(df_actifs.to_string(index=False))

# ==========================================
# 3. ÉVALUATION ET ANALYSE DES COEFFICIENTS
# ==========================================

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

# En binaire, la matrice de coefficients est de dimension (1, n_features)
total_coefficients = model.coef_.size
coefficients_actifs = (model.coef_ != 0).sum()

print(f"Accuracy Train : {acc_train * 100:.2f}%")
print(f"Accuracy Test  : {acc_test * 100:.2f}%")
print(
    f"Sélection Lasso : {coefficients_actifs} coefficients actifs sur un total de {total_coefficients}\n"
)

print("Rapport de classification détaillé (Test) :")
print(
    classification_report(
        y_test, y_pred_test, target_names=label_encoder.classes_
    )
)