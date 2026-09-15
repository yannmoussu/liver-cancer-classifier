# Automondor

Projet d'analyse de données radiomiques pour une compétition de classification ou de prédiction de survie en médecine.

## Description

Ce projet vise à analyser des caractéristiques radiomiques extraites d'images médicales afin de construire des modèles prédictifs (classification ou survie) en utilisant des techniques de régularisation (LASSO, régression logistique) et de sélection de caractéristiques. Le travail inclut la prétraitement des données, l'extraction de caractéristiques, la construction de modèles, leur validation et leur visualisation.

## Structure du projet

Les scripts sont organisés par étape du pipeline. `common/validation.py` est le
seul module partagé entre eux (évaluation par k-fold, tracé des résultats).

- `common/` : Utilitaires partagés (`validation.py`).
- `data_processing/` : Préparation des features brutes (`flattening_multislice.py`, `delta_features.py`).
- `feature_selection/` : Analyse de corrélation et sélection de features (`correlation.py`, `selection_features.py`, `regression_correlation.py`, `visualisation_correlation.py`).
- `modeling/` : Entraînement des modèles (baseline, LASSO, grid search, optimisation des poids) — `baseline.py`, `lasso.py`, `gridsearch.py`, `optipoids.py`, `modele_radiologue.py`, `run_group_lasso.py`, `poids_*.py`.
- `visualization/` : Scripts de visualisation des données et des résultats.
- `experiments/` : Variantes exploratoires (LightGBM/XGBoost/SVM, group LASSO, validations alternatives).
- `notebooks/` : Notebooks Jupyter (préparation, matching, PCA, visualisations).
- `data/` : **Non inclus dans ce dépôt** (voir section [Données](#données) ci-dessous).
- `plots/` : Graphiques générés lors des expérimentations.

## Principaux fichiers

- `modeling/poids_pseudo_radiologue.py` : Méta-classifieur à base de pseudo-features, combinant pseudo-features et features globales en LASSO (`--weights {optimises,correlation} --mode {seuls,lasso}`).
- `modeling/lasso.py` : Implémentation de base de la régression LASSO.
- `common/validation.py` : Fonctions de validation croisée et d'évaluation des modèles.
- `feature_selection/correlation.py` : Analyse de corrélation entre features.
- `data_processing/flattening_multislice.py` / `notebooks/flattening.ipynb` : Processus de flattening des features multislices.
- `notebooks/matching.ipynb` : Appariement des données d'imagerie et cliniques.
- `notebooks/pca_flattened.ipynb` : Analyse en composantes principales (PCA) sur les features aplaties.
- `notebooks/visualisation_clem.ipynb` : Visualisations diverses des résultats.
- `modeling/baseline.py` : Modèle de référence.
- `notebooks/*.ipynb` : Notebooks d'exploration et de production.

## Dépendances

Les principaux packages Python utilisés sont :

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter (pour les notebooks)

Vous pouvez les installer via pip :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Utilisation

### Exécution des scripts Python

Les scripts peuvent être lancés directement depuis la ligne de commande,
**depuis la racine du dépôt** (les chemins vers `data/` sont relatifs à la racine) :

```bash
python modeling/poids_pseudo_radiologue.py
```

Certains scripts peuvent nécessiter des arguments ou des chemins de fichiers spécifiques ; consultez l'en-tête de chaque script pour plus de détails.

### Exécution des notebooks Jupyter

Lancer Jupyter Notebook dans le répertoire du projet :

```bash
jupyter notebook
```

Puis ouvrir les notebooks d'intérêt dans `notebooks/` (par exemple `flattening.ipynb`,
`matching.ipynb`, etc.) et exécuter les cellules étape par étape. La première cellule
de chaque notebook se replace automatiquement à la racine du dépôt, donc peu importe
le répertoire de travail avec lequel Jupyter démarre.

## Résultats

Les sorties comprennent :

- Fichiers CSV de caractéristiques sélectionnées et de scores.
- Modèles entraînés (sauvegardés sous forme de fichiers `.pkl`).
- Graphiques enregistrés dans le dossier `plots/` (courbes ROC, matrice de confusion, distributions, etc.).
- Rapports de classification dans la console ou dans des fichiers logs.

## Données

Ce dépôt est une version publique, préparée à des fins de portfolio, d'un projet
mené en groupe. Le dossier `data/` n'est **pas inclus** : il contenait des
données radiomiques et cliniques de patients réelles (bien qu'anonymisées),
dont la publication n'a pas été autorisée par l'établissement encadrant le
projet. `data/` est donc listé dans `.gitignore`.

À titre indicatif, ce dossier aurait normalement contenu :

- Les features radiomiques extraites des images médicales (par lésion / slice).
- Les métadonnées cliniques des patients (labels, variables démographiques, etc.).
- Les fichiers CSV intermédiaires générés par les étapes de prétraitement,
  d'aplatissement (`flattening`) et d'appariement (`matching`).

Pour exécuter les scripts ou notebooks de ce dépôt, il faut disposer de ces
mêmes fichiers sous `data/` (mêmes noms et structure que ceux référencés dans
le code) ; sans eux, les scripts échoueront à l'étape de lecture des données.

## Notes

- Les chemins d'accès aux fichiers de données sont souvent codés en dur dans les scripts (relatifs à la racine du dépôt, ex. `data/...`) ; lancez toujours les scripts depuis la racine du dépôt.
- Les jeux de données proviennent d'études médicales réelles et sont anonymisés.
- Ce projet est destiné à des fins de recherche et de démonstration.

## Auteurs

Yann MOUSSU, Mathéo CAHITTE, Ilias BOURHRARA, Antonin RICOCHON & Clément COURNIL--RABEUX

## Licence

Le code de ce projet est distribué sous licence MIT (voir [LICENSE](LICENSE)).
Les jeux de données du dossier `data/` sont issus d'une étude médicale réelle
(anonymisée) et ne sont pas couverts par cette licence — ils ne sont pas
publiés dans ce dépôt (voir section [Données](#données)).