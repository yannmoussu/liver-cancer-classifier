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
    print(f"🚀 Lancement du Stratified GROUP K-Fold ({n_splits} Folds)")
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
    
    print(f"⚖️ Équilibrage automatique appliqué sur : {description_equilibrage}")
    # =========================================================================
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    fold_metrics = {'accuracy': [], 'auc': [], 'f1': [], 'mcc': []}
    y_vrais_total, y_pred_total, y_prob_total = [], [], []
    
    repartition_data = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y=strat_array, groups=groups), 1):
        
        # On calcule les répartitions pour chaque variable stratifiée séparément
        for col in df_temp.columns:
            valeurs_uniques, comptes = np.unique(df_temp[col].iloc[test_idx], return_counts=True)
            for val, count in zip(valeurs_uniques, comptes):
                repartition_data.append({'Fold': f"Fold {fold}", 'Variable': col, 'Classe': val, 'Patients': count})
        
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
    print("🏆 BILAN DES MÉTRIQUES ESSENTIELLES (MOYENNE ± ÉCART-TYPE)")
    print("="*50)
    print(f"Accuracy : {np.mean(fold_metrics['accuracy']):.3f} (± {np.std(fold_metrics['accuracy']):.3f})")
    print(f"ROC AUC  : {np.mean(fold_metrics['auc']):.3f} (± {np.std(fold_metrics['auc']):.3f})")
    print(f"F1-Score : {np.mean(fold_metrics['f1']):.3f} (± {np.std(fold_metrics['f1']):.3f})")
    print(f"MCC      : {np.mean(fold_metrics['mcc']):.3f} (± {np.std(fold_metrics['mcc']):.3f})")
    print("="*50)
    
    df_repartition = pd.DataFrame(repartition_data)
    variables = df_temp.columns.tolist()
    n_vars = len(variables)

    # Création des sous-graphiques : un par variable + 1 pour la matrice de confusion
    fig, axes = plt.subplots(1, n_vars + 1, figsize=(4 * (n_vars + 1), 5))
    
    # Sécurisation si jamais 1 seule variable
    if n_vars + 1 == 1:
        axes = [axes]

    # Définition de palettes de couleurs séquentielles (ton sur ton) pour chaque graphe
    base_colors = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds']

    for i, var in enumerate(variables):
        df_var = df_repartition[df_repartition['Variable'] == var]
        df_pivot = df_var.pivot(index='Fold', columns='Classe', values='Patients').fillna(0)
        df_pourcentages = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100
        
        # On génère des couleurs plus intenses en évitant les premières (qui sont presque blanches)
        n_classes = len(df_pourcentages.columns)
        # On demande 2 couleurs supplémentaires et on sélectionne les plus intenses à la fin
        palette = sns.color_palette(base_colors[i % len(base_colors)], n_colors=n_classes + 2)[2:]
        
        df_pourcentages.plot(kind='bar', stacked=True, ax=axes[i], color=palette, edgecolor='white')
        axes[i].set_title(f"Proportions - {var}", fontsize=11, fontweight='bold')
        if i == 0:
            axes[i].set_ylabel("Proportion (%)")
        else:
            axes[i].set_ylabel("")
        axes[i].set_xlabel("")
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        axes[i].tick_params(axis='x', rotation=45)

    ax_cm = axes[-1]
    cm_total = confusion_matrix(y_vrais_total, y_pred_total)
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=noms_classes, yticklabels=noms_classes)
    ax_cm.set_title(f"Confusion Cumulée ({n_splits} Folds)", fontsize=11, fontweight='bold')
    ax_cm.set_ylabel('Vérité Terrain')
    ax_cm.set_xlabel('Prédiction Modèle')
    
    plt.tight_layout()
    
    # --- SAUVEGARDE DU GRAPHIQUE ---
    import os
    from datetime import datetime
    os.makedirs(os.path.join("plots", "regression"), exist_ok=True)
    model_name = type(modele).__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("plots", "regression", f"validation_{model_name}_{timestamp}.png")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📈 Graphique sauvegardé sous : {filename}")
    # -------------------------------
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return y_vrais_total, y_prob_total, fold_metrics