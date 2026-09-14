"""
Action : 
Isoler uniquement les features First Order  d'une seule phase temporelle 
(la plus complète, souvent la phase veineuse).
Appliquer une standardisation robuste et entraîner une Régression Logistique 
sans pénalité complexe (juste L2 faible).  
Livrable : Un score ROC-AUC de référence (baseline) et les poids des coefficients.

LightGBM, SVM
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("data/global_excel_resampled_scaled.csv", sep=";")
df = df[df["classe_name"] != "Mixtes"]
cols_to_keep = [c for c in df.columns if 'firstorder' in c] + ['temps_inj', 'classe_name', 'id']
df_firstorder = df[cols_to_keep]

df_firstorder_all = df_firstorder.dropna()
df_firstorder_port = df_firstorder[df_firstorder['temps_inj'] == 'PORT'].dropna()
df_firstorder_art = df_firstorder[df_firstorder['temps_inj'] == 'ART'].dropna()
df_firstorder_tard = df_firstorder[df_firstorder['temps_inj'] == 'TARD'].dropna()
#df_firstorder_alltimes = df_firstorder.merge(df_firstorder, df_firstorder, on='id', how='inner').dropna()

X_firstorder_all = df_firstorder_all.drop(['id', 'classe_name', 'temps_inj'], axis=1)
X_firstorder_port = df_firstorder_port.drop(['id', 'classe_name', 'temps_inj'], axis=1)
X_firstorder_art = df_firstorder_art.drop(['id', 'classe_name', 'temps_inj'], axis=1)
X_firstorder_tard = df_firstorder_tard.drop(['id', 'classe_name', 'temps_inj'], axis=1)
#X_firstorder_alltimes = df_firstorder_alltimes.drop(['id', 'classe_name'], axis=1)

y_firstorder_all = df_firstorder_all['classe_name']
y_firstorder_port = df_firstorder_port['classe_name']
y_firstorder_art = df_firstorder_art['classe_name']
y_firstorder_tard = df_firstorder_tard['classe_name']
#y_firstorder_alltimes = df_firstorder_alltimes['classe_name']

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_firstorder_all, y_firstorder_all, test_size=0.2)
X_train_port, X_test_port, y_train_port, y_test_port = train_test_split(X_firstorder_port, y_firstorder_port, test_size=0.2)
X_train_art, X_test_art, y_train_art, y_test_art = train_test_split(X_firstorder_art, y_firstorder_art, test_size=0.2)
X_train_tard, X_test_tard, y_train_tard, y_test_tard = train_test_split(X_firstorder_tard, y_firstorder_tard, test_size=0.2)
#X_train_alltimes, X_test_alltimes, y_train_alltimes, y_test_alltimes = train_test_split(X_firstorder_alltimes, y_firstorder_alltimes, test_size=0.2)

feature_names = X_train_port.columns

model_all = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.05)
model_all.fit(X_train_all, y_train_all)
predictions_all = model_all.predict(X_test_all)
probas_all = model_all.predict_proba(X_test_all)
cm_all = confusion_matrix(y_test_all, predictions_all)
display_all = ConfusionMatrixDisplay(cm_all)
display_all.plot()
plt.show()
coef_df = pd.DataFrame({'Feature': feature_names,'Coefficient': model_all.coef_[0]})
coef_df.sort_values('Coefficient').plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 6))
plt.title('Coefficients les plus influents all')
plt.show()

"""
model_port = LogisticRegression()
model_port.fit(X_train_port, y_train_port)
predictions_port = model_port.predict(X_test_port)
probas_port = model_port.predict_proba(X_test_port)
cm_port = confusion_matrix(y_test_port, predictions_port)
display_port = ConfusionMatrixDisplay(cm_port)
display_port.plot()
plt.show()
coef_df = pd.DataFrame({'Feature': feature_names,'Coefficient': model_port.coef_[0]})
coef_df.sort_values('Coefficient').plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 6))
plt.title('Coefficients les plus influents port')
plt.show()

model_art = LogisticRegression()
model_art.fit(X_train_art, y_train_art)
predictions_art = model_art.predict(X_test_art)
probas_art = model_art.predict_proba(X_test_art)
cm_art = confusion_matrix(y_test_art, predictions_art)
display_art = ConfusionMatrixDisplay(cm_art)
display_art.plot()
plt.show()
coef_df = pd.DataFrame({'Feature': feature_names,'Coefficient': model_art.coef_[0]})
coef_df.sort_values('Coefficient').plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 6))
plt.title('Coefficients les plus influents art')
plt.show()

model_tard = LogisticRegression()
model_tard.fit(X_train_tard, y_train_tard)
predictions_tard = model_tard.predict(X_test_tard)
probas_tard = model_tard.predict_proba(X_test_tard)
cm_tard = confusion_matrix(y_test_tard, predictions_tard)
display_tard = ConfusionMatrixDisplay(cm_tard)
display_tard.plot()
plt.show()
coef_df = pd.DataFrame({'Feature': feature_names,'Coefficient': model_tard.coef_[0]})
coef_df.sort_values('Coefficient').plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 6))
plt.title('Coefficients les plus influents tard')
plt.show()
"""
# model_alltimes = LogisticRegression()
# model_alltimes.fit(X_train_alltimes, y_train_alltimes)
# predictions_alltimes = model_alltimes.predict(X_test_alltimes)
# probas_alltimes = model_alltimes.predict_proba(X_test_alltimes)
# cm_alltimes = confusion_matrix(y_test_alltimes, predictions_alltimes)
# display_alltimes = ConfusionMatrixDisplay(cm_alltimes)
# display_alltimes.plot()
# plt.show()
# coef_df = pd.DataFrame({'Feature': feature_names,'Coefficient': model_alltimes.coef_[0]})
# coef_df.sort_values('Coefficient').plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 6))
# plt.title('Coefficients les plus influents alltimes')
# plt.show()