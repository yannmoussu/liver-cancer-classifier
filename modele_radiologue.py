#!/usr/bin/env python3
"""
Modele de classification des tumeurs basé sur les evaluations radiologues.
Objectif: classification dure (logistic regression) entre CHC (type 1) et CCk (type 2)
à partir du fichier Relectures_imageries(Feuil1).csv.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from validation import evaluer_modele_kfold

def load_data(csv_path):
    """Load the CSV file with semicolon separator."""
    df = pd.read_csv(csv_path, sep=';')
    return df

def preprocess(df):
    """Select features and target, handle missing values."""
    # Keep only CHC (1) and CK (2)
    df = df[df['Type_tumeur'].isin([1, 2])].copy()
    # Define target
    y = df['Type_tumeur'].map({1: 0, 2: 1})  # 0 for CHC, 1 for CCk
    # Feature columns (exclude patient number and target)
    feature_cols = [col for col in df.columns if col not in ['Type_tumeur', 'Patient_number']]
    X = df[feature_cols]
    # Replace empty strings with NaN then fill with column mean (or 0 for binary?)
    X = X.replace('', np.nan)
    # For simplicity, fill NaN with 0 (assuming missing indicates absence)
    X = X.fillna(0)
    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, y, feature_cols, df

def main():
    csv_path = 'data/Relectures_imageries(Feuil1).csv'
    df_raw = load_data(csv_path)
    print(f"Dataset shape: {df_raw.shape}")
    print(f"Tumor type distribution:\n{df_raw['Type_tumeur'].value_counts()}")

    # Preprocess (filtering inside)
    X, y, feature_names, df = preprocess(df_raw)
    print(f"Features shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = logreg.predict(X_test_scaled)
    y_pred_proba = logreg.predict_proba(X_test_scaled)

    print("\n=== Logistic Regression Results ===")
    print("Accuracy:", logreg.score(X_test_scaled, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['CHC', 'CCk']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance (coefficients)
    coefs = logreg.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs)
    }).sort_values('abs_coefficient', ascending=False)

    print("\n=== Feature Importance (absolute coefficient) ===")
    print(feature_importance[['feature', 'coefficient']])

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(15), x='coefficient', y='feature', hue='feature', palette='viridis', legend=False)
    plt.title('Top 15 Features - Logistic Regression Coefficients')
    plt.xlabel('Coefficient')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/logreg_feature_importance_Relectures_imagerie.png'
    plt.savefig(plot_path)
    print(f"\nSaved feature importance plot to {plot_path}")

    # PCA for visualization (optional)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'type': y_train.map({0: 'CHC', 1: 'CCk'})
    })
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='type', palette='Set1', s=60, alpha=0.7)
    plt.title('PCA of Training Set (CHC vs CCk)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/pca_chc_cck_Relectures_imagerie.png'
    plt.savefig(plot_path)
    print(f"Saved PCA plot to {plot_path}")

    # NEW: PCA with decision boundary
    # Fit logistic regression on the 2 PCA components
    logreg_pca = LogisticRegression(random_state=42, max_iter=1000)
    logreg_pca.fit(X_pca, y_train)
    # Create a mesh to plot decision boundary
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = logreg_pca.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs, levels=25, cmap='RdBu', alpha=0.3)
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='type', palette='Set1', s=60, alpha=0.8, edgecolor='k')
    plt.title('PCA with Logistic Regression Decision Boundary (CHC vs CCk)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/pca_chc_cck_decision_boundary_Relectures_imagerie.png'
    plt.savefig(plot_path)
    print(f"Saved PCA decision boundary plot to {plot_path}")

    # NEW: Explained variance plot (scree plot)
    pca_full = PCA().fit(X_train_scaled)
    explained_var = pca_full.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    components = range(1, len(explained_var) + 1)
    plt.bar(components, explained_var, alpha=0.7, color='steelblue')
    plt.step(components, np.cumsum(explained_var), where='mid',
             label='Cumulative explained variance', color='orange', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label='95% threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance (Scree Plot)')
    plt.xticks(list(components))
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/pca_explained_variance_Relectures_imagerie.png'
    plt.savefig(plot_path)
    print(f"Saved explained variance plot to {plot_path}")

    # NEW: Correlation matrix heatmap of original features
    # Use X (original features after preprocessing) as DataFrame for correlation
    X_df = pd.DataFrame(X, columns=feature_names)
    corr = X_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/correlation_matrix_Relectures_imagerie.png'
    plt.savefig(plot_path)
    print(f"Saved correlation matrix plot to {plot_path}")

    # Evaluate with Stratified Group K-Fold using patient number as group
    # We use the filtered df (df) and group by Patient_number to avoid leakage
    print("\n=== Cross-validation with Stratified Group K-Fold (by Patient) ===")
    evaluer_modele_kfold(
        logreg,
        X,
        y,
        df,  # filtered dataframe (only CHC and CCk)
        ['Patient_number'],  # group by patient to avoid intra-patient leakage
        ['CHC', 'CCk']
    )

    # Optionally, save model and scaler for later use on mixte tumors
    import joblib
    joblib.dump(logreg, 'logreg_chc_cck.pkl')
    joblib.dump(scaler, 'scaler_chc_cck.pkl')
    print("Saved model and scaler to disk.")

if __name__ == '__main__':
    main()