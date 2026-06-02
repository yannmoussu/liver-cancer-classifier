import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    matthews_corrcoef, 
    confusion_matrix
)

def evaluer_modele_kfold(modele, X, y, classes_names, n_splits=5, random_seed=42):
    print(f"🚀 Lancement de la validation croisée stricte ({n_splits} Folds)")
    np.random.seed(random_seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    fold_metrics = {
        'accuracy': [], 'auc': [], 'f1': [], 'mcc': []
    }
    y_vrais_total, y_pred_total, y_prob_total = [], [], []

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
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['auc'].append(auc)
        fold_metrics['f1'].append(f1)
        fold_metrics['mcc'].append(mcc)

        print(f"Fold {fold}/{n_splits} - Accuracy: {acc:.2f} | AUC: {auc:.2f} | F1: {f1:.2f} | MCC: {mcc:.2f}")

    print("\n" + "="*50)
    print("🏆 BILAN DES MÉTRIQUES ESSENTIELLES (MOYENNE ± ÉCART-TYPE)")
    print("="*50)
    print(f"Accuracy (Précision globale) : {np.mean(fold_metrics['accuracy']):.3f} (± {np.std(fold_metrics['accuracy']):.3f})")
    print(f"ROC AUC (Pouvoir séparateur): {np.mean(fold_metrics['auc']):.3f} (± {np.std(fold_metrics['auc']):.3f})")
    print(f"F1-Score (Équilibre global)  : {np.mean(fold_metrics['f1']):.3f} (± {np.std(fold_metrics['f1']):.3f})")
    print(f"Matthews Correlation (MCC)   : {np.mean(fold_metrics['mcc']):.3f} (± {np.std(fold_metrics['mcc']):.3f})")
    print("="*50)
    
    plt.figure(figsize=(5, 4))
    cm_total = confusion_matrix(y_vrais_total, y_pred_total)
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_names, yticklabels=classes_names)
    plt.title(f"Matrice de Confusion Cumulée ({n_splits} Folds)")
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction Modèle')
    plt.show()
    
    return y_vrais_total, y_prob_total