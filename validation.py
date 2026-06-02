import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    matthews_corrcoef, 
    confusion_matrix
)

def evaluer_modele_kfold(
    modele, 
    X,                  # La matrice des caractéristiques (X_final)
    y,                  # Le vecteur cible (y_final)
    data,               # Le DataFrame contenant les métadonnées cliniques
    col_group,          # Nom de la colonne OU Liste de colonnes (ex: ['patient_id', 'lesion_id'])
    noms_classes,       # Liste des noms pour l'affichage ['CHC', 'CCK']
    col_age=None,       # (Optionnel) Nom de la colonne Âge
    col_sexe=None,      # (Optionnel) Nom de la colonne Sexe
    n_splits=5, 
    random_seed=42
):
    print(f"🚀 Lancement du Stratified GROUP K-Fold ({n_splits} Folds)")
    np.random.seed(random_seed)
    
    if isinstance(col_group, list):

        groups = data[col_group].astype(str).agg('_'.join, axis=1).to_numpy()
    else:
        groups = data[col_group].to_numpy()
        

    elements_stratification = [pd.Series(y).astype(str)]
    description_equilibrage = "Maladie"
    
    if col_sexe is not None:
        sexe_vals = data[col_sexe].to_numpy()
        elements_stratification.append(pd.Series(sexe_vals).astype(str))
        description_equilibrage += f" + Sexe ('{col_sexe}')"
        
    if col_age is not None:
        age_vals = data[col_age].to_numpy()
        tranches_age = pd.cut(age_vals, bins=[0, 50, 70, 120], labels=['<50', '50-70', '>70']).astype(str)
        elements_stratification.append(pd.Series(tranches_age))
        description_equilibrage += f" + Âge ('{col_age}')"
    
    df_temp = pd.concat(elements_stratification, axis=1)
    strat_array = df_temp.apply(lambda row: "_".join(row), axis=1).to_numpy()
    
    print(f"⚖️ Équilibrage automatique appliqué sur : {description_equilibrage}")
    # =========================================================================
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    fold_metrics = {'accuracy': [], 'auc': [], 'f1': [], 'mcc': []}
    y_vrais_total, y_pred_total, y_prob_total = [], [], []
    
    repartition_data = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y=strat_array, groups=groups), 1):
        
        valeurs_uniques, comptes = np.unique(strat_array[test_idx], return_counts=True)
        for val, count in zip(valeurs_uniques, comptes):
            repartition_data.append({'Fold': f"Fold {fold}", 'Sous-groupe': val, 'Patients': count})
        
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
    print(f"Accuracy : {np.mean(fold_metrics['accuracy']):.3f} (± {np.std(fold_metrics['accuracy']):.3f})")
    print(f"ROC AUC  : {np.mean(fold_metrics['auc']):.3f} (± {np.std(fold_metrics['auc']):.3f})")
    print(f"F1-Score : {np.mean(fold_metrics['f1']):.3f} (± {np.std(fold_metrics['f1']):.3f})")
    print(f"MCC      : {np.mean(fold_metrics['mcc']):.3f} (± {np.std(fold_metrics['mcc']):.3f})")
    print("="*50)
    
    df_repartition = pd.DataFrame(repartition_data)
    df_pivot = df_repartition.pivot(index='Fold', columns='Sous-groupe', values='Patients').fillna(0)
    df_pourcentages = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df_pourcentages.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab20', edgecolor='white')
    axes[0].set_title(f"Composition des Folds de Test\n({description_equilibrage})", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Proportion (%)")
    axes[0].set_xlabel("")
    axes[0].legend(title="Sous-groupes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    axes[0].tick_params(axis='x', rotation=0)

    cm_total = confusion_matrix(y_vrais_total, y_pred_total)
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
                xticklabels=noms_classes, yticklabels=noms_classes)
    axes[1].set_title(f"Matrice de Confusion Cumulée ({n_splits} Folds)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Vérité Terrain')
    axes[1].set_xlabel('Prédiction Modèle')
    
    plt.tight_layout()
    plt.show()
    
    return y_vrais_total, y_prob_total