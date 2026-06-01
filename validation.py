import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.base import clone

def evaluer_modele_kfold(modele, X, y, classes_names, n_splits=5, random_seed=42):
    
    np.random.seed(random_seed)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    fold_metrics = {'accuracy': [], 'auc': []}
    
    y_vrais_total = []
    y_pred_total = []
    y_prob_total = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        modele_fold = clone(modele)
        
        modele_fold.fit(X_train, y_train)
        
        y_pred = modele_fold.predict(X_test)
        y_prob = modele_fold.predict_proba(X_test)[:, 1] 
        

        y_vrais_total.extend(y_test)
        y_pred_total.extend(y_pred)
        y_prob_total.extend(y_prob)
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        fold_metrics['accuracy'].append(acc)
        fold_metrics['auc'].append(auc)
        
        print(f"Fold {fold}/{n_splits} - Précision Globale (Acc): {acc:.2f} | Score AUC: {auc:.2f}")

    print("\n" + "="*40)
    print("BILAN GLOBAL SUR TOUS LES FOLDS")
    print("="*40)
    print(f"Accuracy Moyenne : {np.mean(fold_metrics['accuracy']):.3f} (± {np.std(fold_metrics['accuracy']):.3f})")
    print(f"ROC AUC Moyen    : {np.mean(fold_metrics['auc']):.3f} (± {np.std(fold_metrics['auc']):.3f})")
    
    plt.figure(figsize=(6, 5))
    cm_total = confusion_matrix(y_vrais_total, y_pred_total)
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_names, yticklabels=classes_names)
    plt.title(f"Matrice de Confusion Cumulée (Les {n_splits} Folds)")
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction Modèle')
    plt.show()
    
    return y_vrais_total, y_prob_total