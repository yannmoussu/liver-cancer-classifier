import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import webbrowser
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
except ImportError:
    px = None


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    df = df.rename(columns=lambda c: c.strip())
    df = df.replace({'': np.nan})

    feature_columns = [
        col for col in df.columns
        if not col.startswith('diagnostics_') and col not in {'classe_name', 'patient_num'}
    ]

    for col in feature_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .replace({'nan': np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, feature_columns


def select_top_features_by_variance(df: pd.DataFrame, feature_columns: List[str], top_n: int) -> List[str]:
    variances = df[feature_columns].var(skipna=True).sort_values(ascending=False)
    return variances.head(top_n).index.tolist()


def remove_outliers_iqr(df: pd.DataFrame, features: List[str], factor: float = 1.5) -> pd.DataFrame:
    filtered = df.copy()
    for feature in features:
        if feature not in filtered.columns:
            continue
        col = filtered[feature]
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        filtered = filtered[(col >= lower) & (col <= upper) | col.isna()]
    return filtered


def plot_feature_distributions_by_class(df: pd.DataFrame, features: List[str], output_dir: Path) -> None:
    plot_data = df[['classe_name'] + features].dropna(subset=['classe_name'])
    if plot_data.empty:
        print('Aucune donnée de classe valide pour les distributions de features.')
        return

    for feature in features:
        feature_data = plot_data[['classe_name', feature]].dropna(subset=[feature])
        if feature_data.empty:
            print(f'Aucune donnée numérique disponible pour la feature {feature}.')
            continue

        sns.set(style='whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        sns.boxplot(
            data=feature_data,
            x='classe_name',
            y=feature,
            palette='Set3',
            ax=ax,
            showfliers=False
        )
        sns.stripplot(
            data=feature_data,
            x='classe_name',
            y=feature,
            color='black',
            size=3,
            alpha=0.5,
            jitter=True,
            ax=ax
        )
        ax.set_title(f'Distribution de {feature} par classe')
        ax.set_xlabel('Classe de cancer du foie')
        ax.set_ylabel(feature)
        ax.tick_params(axis='x', rotation=45)

        output_path = output_dir / f'{feature}_distribution_by_class.png'
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f'Enregistré : {output_path}')


def plot_class_feature_heatmap(df: pd.DataFrame, features: List[str], output_dir: Path) -> None:
    plot_data = df.dropna(subset=['classe_name'])
    if plot_data.empty:
        print('Aucune donnée de classe valide pour la heatmap de features.')
        return

    means = plot_data.groupby('classe_name')[features].mean()
    if means.empty:
        print('Aucune moyenne de feature disponible pour la heatmap.')
        return

    fig, ax = plt.subplots(figsize=(max(8, len(features) * 0.6), max(6, len(means) * 0.5)), constrained_layout=True)
    sns.heatmap(means, cmap='vlag', center=means.values.mean(), annot=False, fmt='.2f', ax=ax)
    ax.set_title('Moyennes des features par classe')
    ax.set_xlabel('Features')
    ax.set_ylabel('Classe de cancer du foie')

    output_path = output_dir / 'class_feature_mean_heatmap.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Enregistré : {output_path}')


def plot_bivariate_visualization(df: pd.DataFrame, features: List[str], output_dir: Path) -> None:
    plot_data = df.dropna(subset=['classe_name'] + features)
    if plot_data.empty:
        print('Aucune donnée valide pour la visualisation bivariée.')
        return

    selected = features[:5]
    plot_data = plot_data[['classe_name'] + selected]

    sns.set(style='whitegrid')
    pairplot = sns.pairplot(
        plot_data,
        hue='classe_name',
        palette='tab10',
        diag_kind='kde',
        plot_kws={'alpha': 0.7, 's': 40},
        corner=False
    )
    pairplot.fig.suptitle('Visualisation bivariée des principales features par classe', y=1.02)
    output_path = output_dir / 'bivariate_pairplot.png'
    pairplot.savefig(output_path, dpi=150)
    plt.close('all')
    print(f'Enregistré : {output_path}')

    corr = plot_data[selected].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(selected) * 1.2), max(6, len(selected) * 1.2)), constrained_layout=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Corrélation bivariée des principales features')
    output_path = output_dir / 'bivariate_feature_correlation_heatmap.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Enregistré : {output_path}')


def normalize_features(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def plot_pca(df: pd.DataFrame, features: List[str], output_dir: Path, show_html: bool = False) -> None:
    plot_data = df.dropna(subset=['classe_name'])
    if plot_data.empty:
        print('Aucune donnée valide pour la PCA.')
        return

    X = plot_data[features].copy()
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_scaled = normalize_features(X_imputed)

    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        components,
        columns=[f'PC{i+1}' for i in range(components.shape[1])],
        index=plot_data.index
    )
    pca_df['classe_name'] = plot_data['classe_name'].values

    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='classe_name',
        palette='tab10',
        alpha=0.8,
        s=80,
        edgecolor='w',
        ax=ax
    )
    ax.set_title('PCA sur données normalisées : PC1 vs PC2')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    ax.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')

    output_path = output_dir / 'pca_pc1_pc2.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Enregistré : {output_path}')

    if components.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        classes = pca_df['classe_name'].unique()
        palette = sns.color_palette('tab10', n_colors=len(classes))
        class_palette = dict(zip(classes, palette))

        for cls in classes:
            cls_data = pca_df[pca_df['classe_name'] == cls]
            ax.scatter(
                cls_data['PC1'],
                cls_data['PC2'],
                cls_data['PC3'],
                label=cls,
                color=class_palette[cls],
                s=50,
                alpha=0.8,
                edgecolor='w'
            )

        ax.set_title('PCA 3D des features non diagnostics PC1 vs PC2 vs PC3')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}% variance)')
        ax.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')

        output_path = output_dir / 'pca_pc1_pc2_pc3.png'
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f'Enregistré : {output_path}')

        plot_pca_interactive_3d(pca_df, pca, output_dir, show_html=show_html)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(
        [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        pca.explained_variance_ratio_ * 100,
        color='#4C72B0'
    )
    ax.set_title('Variance expliquée par les composantes principales')
    ax.set_ylabel('Pourcentage de variance expliquée')
    ax.set_xlabel('Composante principale')
    ax.set_ylim(0, max(pca.explained_variance_ratio_ * 100) * 1.1)

    for i, value in enumerate(pca.explained_variance_ratio_ * 100):
        ax.text(i, value + 0.5, f'{value:.1f}%', ha='center')

    output_path = output_dir / 'pca_explained_variance.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Enregistré : {output_path}')

    if components.shape[1] >= 2:
        loadings = pd.DataFrame(
            pca.components_[:2].T,
            index=features,
            columns=['PC1', 'PC2']
        )
        top_loadings = loadings.abs().sum(axis=1).sort_values(ascending=False).head(15).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        sns.heatmap(loadings.loc[top_loadings], annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Charges des features sur PC1 et PC2')
        output_path = output_dir / 'pca_loadings_top_features.png'
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f'Enregistré : {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualiser les features non diagnostics par classe et calculer une PCA pour le fichier global_excel_resampled(Sheet1).csv'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path('data/global_excel_resampled(Sheet1).csv'),
        help='Chemin vers le fichier CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/plots_global'),
        help='Dossier de sortie pour les images'
    )
    parser.add_argument(
        '--top-features',
        type=int,
        default=12,
        help='Nombre de features non diagnostics à utiliser pour les graphiques et la PCA'
    )
    parser.add_argument(
        '--outlier-factor',
        type=float,
        default=1.5,
        help='Facteur IQR pour le filtrage des valeurs aberrantes'
    )
    parser.add_argument('--show', action='store_true', help='Afficher les figures à l\'écran')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, feature_columns = load_data(args.csv)
    selected_features = select_top_features_by_variance(df, feature_columns, args.top_features)

    print('Features utilisées pour les distributions et la PCA :')
    for feature in selected_features:
        print(f' - {feature}')

    df_clean = remove_outliers_iqr(df, selected_features, factor=args.outlier_factor)
    print(f'Données nettoyées : {len(df) - len(df_clean)} lignes supprimées pour aberration (IQR={args.outlier_factor}).')

    plot_feature_distributions_by_class(df_clean, selected_features, args.output_dir)
    plot_class_feature_heatmap(df_clean, selected_features, args.output_dir)
    plot_bivariate_visualization(df_clean, selected_features, args.output_dir)
    plot_pca(df_clean, selected_features, args.output_dir, show_html=args.show)

    if args.show:
        plt.show()


def plot_pca_interactive_3d(pca_df: pd.DataFrame, pca: PCA, output_dir: Path, show_html: bool = False) -> None:
    if px is None:
        print('Plotly n\'est pas installé : impossible de générer la visualisation 3D interactive.')
        return

    if 'PC3' not in pca_df.columns:
        print('Pas assez de composantes pour la visualisation 3D interactive.')
        return

    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='classe_name',
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)',
            'PC3': f'PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}% variance)',
            'classe_name': 'Classe'
        },
        title='PCA 3D interactive des features non diagnostics',
        width=1000,
        height=800
    )
    fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))

    output_path = output_dir / 'pca_pc1_pc2_pc3_interactive.html'
    fig.write_html(str(output_path))
    print(f'Enregistré : {output_path}')

    if show_html:
        webbrowser.open(output_path.as_uri())


if __name__ == '__main__':
    main()
