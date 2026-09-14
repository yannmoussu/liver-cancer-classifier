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
    random_seed=42,
    show_plots=True
):
    print(f"Lancement du Stratified GROUP K-Fold ({n_splits} Folds)")
    np.random.seed(random_seed)
    
    if isinstance(col_group, list):
        groups = data[col_group].astype(str).agg('_'.join, axis=1).to_numpy()
    else:
        groups = data[col_group].to_numpy()
        

    # On s'assure que y est un tableau numpy pour éviter les décalages d'index avec pd.Series
    y_array = np.array(y)
    
    # Remplacement des valeurs par le nom des classes pour une légende plus parlante
    def get_class_name(val):
        try:
            return noms_classes[int(float(val))]
        except:
            return str(val)
            
    y_str = pd.Series(y_array, name="Maladie").map(get_class_name)
    elements_stratification = [y_str]
    description_equilibrage = "Maladie"
    
    if col_sexe is not None:
        sexe_vals = data[col_sexe].to_numpy()
        
        def map_sexe(x):
            try:
                val = float(x)
                if val == 1.0: return "H"
                if val == 0.0: return "F"
                return "H" if val > 0 else "F"
            except:
                return str(x)
                
        sexe_str = pd.Series(sexe_vals, name="Sexe").map(map_sexe)
        elements_stratification.append(sexe_str)
        description_equilibrage += f" + Sexe ('{col_sexe}')"
        
    if col_age is not None:
        age_vals = data[col_age].to_numpy()
        
        # Si la colonne est standardisée (StandardScaler), les valeurs max sont très petites (ex: < 5)
        if np.nanmax(age_vals) < 10:
            # On utilise les écarts-types comme approximations de vos tranches habituelles
            bins = [-np.inf, -0.5, 0.5, np.inf]
            labels = ['<50', '50-70', '>70']
        else:
            bins = [0, 50, 70, 120]
            labels = ['<50', '50-70', '>70']
            
        tranches_age = pd.cut(age_vals, bins=bins, labels=labels)
        # On force la conversion en string et on gère les NaN pour éviter les conflits str/float dans np.unique
        tranches_age = [str(x) if pd.notna(x) else "Inconnu" for x in tranches_age]
        
        elements_stratification.append(pd.Series(tranches_age, name="Âge"))
        description_equilibrage += f" + Âge ('{col_age}')"
    
    df_temp = pd.concat(elements_stratification, axis=1)
    # On utilise un espace pour un affichage propre : "CCK H 50-70"
    strat_array = df_temp.apply(lambda row: " ".join([str(x) for x in row]), axis=1).to_numpy()
    
    print(f"Équilibrage automatique appliqué sur : {description_equilibrage}")
    # =========================================================================
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    fold_metrics = {'accuracy': [], 'auc': [], 'f1': [], 'mcc': []}
    y_vrais_total, y_pred_total, y_prob_total = [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y=strat_array, groups=groups), 1):
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
        f1 = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['auc'].append(auc)
        fold_metrics['f1'].append(f1)
        fold_metrics['mcc'].append(mcc)

        print(f"Fold {fold}/{n_splits} - Accuracy: {acc:.2f} | AUC: {auc:.2f} | F1: {f1:.2f} | MCC: {mcc:.2f}")

    print("\n" + "="*50)
    print("BILAN DES MÉTRIQUES ESSENTIELLES (MOYENNE ± ÉCART-TYPE)")
    print("="*50)
    print(f"Accuracy : {np.mean(fold_metrics['accuracy']):.3f} (± {np.std(fold_metrics['accuracy']):.3f})")
    print(f"ROC AUC  : {np.mean(fold_metrics['auc']):.3f} (± {np.std(fold_metrics['auc']):.3f})")
    print(f"F1-Score : {np.mean(fold_metrics['f1']):.3f} (± {np.std(fold_metrics['f1']):.3f})")
    print(f"MCC      : {np.mean(fold_metrics['mcc']):.3f} (± {np.std(fold_metrics['mcc']):.3f})")
    print("="*50)
    
    # --- CRÉATION DE LA FIGURE UNIQUEMENT AVEC LA MATRICE DE CONFUSION ---
    plt.figure(figsize=(6, 5))
    cm_total = confusion_matrix(y_vrais_total, y_pred_total)
    
    # Configuration de la heatmap en taille 48 de base
    ax_cm = sns.heatmap(
        cm_total, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=noms_classes, 
        yticklabels=noms_classes,
        cbar=False,
        annot_kws={"size": 28, "weight": "bold"}
    )
    
    # Modification de la taille et de la graisse SANS écraser le choix automatique de couleur de Seaborn
    for text in ax_cm.texts:
        text.set_size(28)          
        text.set_weight('bold')    
        
    plt.title(f"Confusion Cumulée ({n_splits} Folds)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Vérité Terrain', fontsize=12, fontweight='bold')
    plt.xlabel('Prédiction Modèle', fontsize=12, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()
    
    # --- SAUVEGARDE DU GRAPHIQUE ---
    import os
    from datetime import datetime
    os.makedirs(os.path.join("plots", "regression"), exist_ok=True)
    model_name = type(modele).__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("plots", "regression", f"validation_{model_name}_{timestamp}.png")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé sous : {filename}")
    # -------------------------------
    
    plt.show()
    
    return y_vrais_total, y_prob_total, fold_metrics