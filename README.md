# Automondor

Projet d'analyse de données radiomiques pour une compétition de classification ou de prédiction de survie en médecine.

## Description

Ce projet vise à analyser des caractéristiques radiomiques extraites d'images médicales afin de construire des modèles prédictifs (classification ou survie) en utilisant des techniques de régularisation (LASSO, régression logistique) et de sélection de caractéristiques. Le travail inclut la prétraitement des données, l'extraction de caractéristiques, la construction de modèles, leur validation et leur visualisation.

## Structure du projet

- `automondor/` : Dossier contenant des scripts et notebooks principaux.
- `data/` : Jeux de données utilisés (features, métadonnées patients, résultats intermédiaires).
- `plots/` : Graphiques générés lors des expérimentations.
- Fichiers Python autonomes : divers traitements et modèles.
- Notebooks Jupyter : analyses interactives et visualisations.

## Principaux fichiers

- `poids_optimisés_lasso.py` : Script principal pour l'optimisation des poids dans un modèle LASSO en combinant pseudo-features et features globales.
- `lasso.py` : Implémentation de base de la régression LASSO.
- `validation.py` : Fonctions de validation croisée et d'évaluation des modèles.
- `correlation.py` : Analyse de corrélation entre features.
- `flattening*.py` / `flattening.ipynb` : Processus de flattening des features multislices.
- `matching.ipynb` : Appariement des données d'imagerie et cliniques.
- `pca_flattened.ipynb` : Analyse en composantes principales (PCA) sur les features aplaties.
- `visualisation_clem.ipynb` : Visualisations diverses des résultats.
- `baseline.py` : Modèle de référence.
- `*.ipynb` : Notebooks d'exploration et de production.

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

Les scripts peuvent être lancés directement depuis la ligne de commande :

```bash
python poids_optimisés_lasso.py
```

Certains scripts peuvent nécessiter des arguments ou des chemins de fichiers spécifiques ; consultez l'en-tête de chaque script pour plus de détails.

### Exécution des notebooks Jupyter

Lancer Jupyter Notebook dans le répertoire du projet :

```bash
jupyter notebook
```

Puis ouvrir les notebooks d'intérêt (par exemple `flattening.ipynb`, `matching.ipynb`, etc.) et exécuter les cellules étape par étape.

## Résultats

Les sorties comprennent :

- Fichiers CSV de caractéristiques sélectionnées et de scores.
- Modèles entraînés (sauvegardés sous forme de fichiers `.pkl`).
- Graphiques enregistrés dans le dossier `plots/` (courbes ROC, matrice de confusion, distributions, etc.).
- Rapports de classification dans la console ou dans des fichiers logs.

## Notes

- Les chemins d'accès aux fichiers de données sont souvent codés en dur dans les scripts ; assurez-vous que le dossier `data/` est présent au même niveau que les scripts.
- Les jeux de données proviennent d'études médicales réelles et sont anonymisés.
- Ce projet est destiné à des fins de recherche et de démonstration.

## Auteurs

Yann MOUSSU, Mathéo CAHITTE, Ilias BOURHRARA, Antonin RICOCHON & Clément COURNIL--RABEUX

## Licence

Le code de ce projet est distribué sous licence MIT (voir [LICENSE](LICENSE)).
Les jeux de données du dossier `data/` sont issus d'une étude médicale réelle
(anonymisée) et ne sont pas couverts par cette licence — leur diffusion est
en attente d'autorisation.