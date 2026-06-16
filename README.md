# MODULE DL2(DEEP LEARNING 2) DEVOIR 3 ENTRE 11-16 JUIN 2026

# Fine-tuner un modèle BERT pour la classification de texte et déployer une démo interactive

> **Dakar Institute of Technology (DIT)  Master 2 Intelligence Artificielle**  
> **Devoir N°3 Deep Learning**  
> **Réalisé par : Alpha Oumar DIALLO & Jean-Fabrice OUFFOUE** 
> Modèle : `google-bert/bert-base-multilingual-cased` | Dataset : InShort News Data | 7 catégories

---
## Table des matières

1. [Objectif](#Objectif)
2. [Résultats obtenus](#résultats-obtenus)
3. [Architecture du projet](#architecture-du-projet)
4. [Installation](#installation)
5. [Données](#données)
6. [Utilisation](#utilisation)
7. [Structure du code](#structure-du-code)
8. [Hyperparamètres](#hyperparamètres)
9. [Approche combinée](#approche-combinée)
10. [Analyse des résultats](#analyse-des-résultats)
11. [Interface de démonstration](#interface-de-démonstration)
12. [Fichiers générés](#fichiers-générés)
13. [Dépendances](#dépendances)

---

## Objectif


Dans ce projet , nous allons **classifier la colonne "news_headline"+"news_article" de notre dataset  suivant la colonne  "news_category"**. Vous trouverez la source de  notre **dataset** à l'addresse suivante :
[lien du dataset](https://drive.google.com/file/d/1S6D_YMvB6W7yggkPG23Rm_u5FV9Z6_DZ/view?usp=drive_link)

Ce projet fine-tune **BERT multilingue** pour classifier automatiquement des articles de presse en 7 catégories thématiques à partir du dataset **InShort News**.

L'approche optimale retenue consiste à concaténer le **titre** et le **corps de l'article** (`combined`) avant de les passer à BERT, ce qui maximise la richesse sémantique de l'entrée et la confiance des prédictions.

```
news_headline + ". " + news_article  →  BERT  →  Catégorie prédite
```

**Classes détectées :** `automobile` · `entertainment` · `politics` · `science` · `sports` · `technology` · `world`




## Structure du projet  

```
.
└── bert-classification-inshort_news_data-1-2.csv
    ├── data
    │   └── inshort_news_data-1 2.csv      <- dataset
    ├── dataset.py                         <- TextClassificationDataset
    ├── demo.py                            <- interface Gradio
    ├── model.py                           <- chargement/définition du modèle BERT
    ├── README.md                          <- rapport (bonus)
    ├── requirements.txt
    ├── train.py                           <- boucles train_epoch/eval_epoch + main
    └── utils.py                           <- métriques, seed, visualisations

```


## Données

### Dataset attendu

Placer le fichier CSV dans `data/` :

```
data/
└── inshort_news_data-1-2.csv
```

Le CSV doit contenir au minimum ces trois colonnes :

| Colonne          | Description            | Exemple                                  |
|------------------|------------------------|------------------------------------------|
| `news_headline`  | Titre de l'article     | *India launches new space mission...*    |
| `news_article`   | Corps de l'article     | *ISRO successfully launched...*          |
| `news_category`  | Catégorie (label cible)| `science`, `sports`, `technology`...     |

> Le chemin et les noms de colonnes sont configurables dans `CONFIG` (fichier `model.py`).

### Nettoyage appliqué

1. Suppression des lignes avec `NaN` dans les 3 colonnes
2. Strip des espaces en début/fin de chaîne
3. Suppression des lignes avec titre ou article vide
4. Concaténation : `text_combined = headline + ". " + article`

### Inspecter le dataset avant d'entraîner

```bash
python train.py --inspect
```

Génère deux graphiques d'analyse :
- `class_distribution_inspection.png` -  distribution et proportions par classe
- `token_length_distribution.png` -  histogrammes des longueurs BERT et boxplots par classe

> **Recommandation :** lancer `--inspect` avant tout entraînement pour vérifier que `max_length=256` couvre bien ≥ 95% des paires titre+article de votre dataset.


### Flux de données

```
CSV  ──►  load_and_inspect_data()  ──►  TextClassificationDataset
                                               │
                                    ┌──────────┴──────────┐
                                 DataLoader            DataLoader
                                  (train)                (val)
                                    │                     │
                          BertForSequenceClassification   │
                          (bert-base-multilingual-cased)  │
                                    │                     │
                             CrossEntropyLoss ◄───────────┘
                                    │
                          AdamW + LinearScheduler
                          (warmup 10% → décroissance)
                                    │
                          Sauvegarde best_model/
                          (critère : val_loss minimale)
```


## Analyse du dataset 

Avant d’écrire la moindre ligne de code, nous inspectons le dataset :

-  Nombre total d'exemples et nombre de classes  
-  Distribution des classes , si déséquilibre > 2:1, justifiez votre stratégie 
-  Longueur des textes (min, max, moyenne en nombre de tokens) , pour choisir max_length
-  Afficher au moins 5 exemples de textes avec leurs labels

voir le repertoire [data-inspect](data-inspect)  pour plus de détails. 




##  Déploiement de nos applications et modèles

Nous avons déployé notre application et modèle sur la plateforme [huggingface](https://huggingface.co/). 
Vous les  touverez via les différents liens ci-dessous:
- [Notre application de classification du titre + contenu d'articles de presse](https://huggingface.co/spaces/jfo25/DL2-Devoir-3-bert-classification-articles-de-presse)
- [Sauvegarde du modèle associé à l'application ci-dessus](https://huggingface.co/jfo25/DL2-Devoir-3-bert-model-classification-articles-de-presse/tree/main)

 ### Fonctionnalités

- Saisie libre du titre et/ou de l'article
- Prédiction instantanée avec score de confiance
- Graphique interactif des probabilités sur toutes les classes
- Indicateur du mode d'entrée (cas idéal ou fallback)

Nous vous informons par ailleurs qu'avant de déployer application ci-dessus, nous avons déployé une version qui permet de classifier uniquement les **titres** des articles(le modèle associé est en en effet entrainé uniquement sur les colonnes `news_headline` et `news_category`). Nous vous donnons également accès à cette dernière [application](https://huggingface.co/spaces/jfo25/classification-titres-articles-de-presse) ainsi qu'au [modèle](https://huggingface.co/jfo25/model-classification-titres-articles-de-presse/tree/main) associé. 



## Outils

Nous utilisons Python 3.9+, PyTorch ≥2.0, transformers (Hugging Face), Gradio, scikit-learn GPU recommandé (Google Colab accepté)

## Dépendances

```
torch>=2.0.0          # Framework deep learning
transformers>=4.40.0  # Modèle BERT + tokenizer HuggingFace
pandas>=2.0.0         # Chargement et manipulation du CSV
scikit-learn>=1.3.0   # Split stratifié, métriques, LabelEncoder
numpy>=1.24.0         # Calculs numériques
matplotlib>=3.7.0     # Visualisations
seaborn>=0.12.0       # Heatmap matrice de confusion
wandb>=0.17.0         # Suivi des expériences
gradio>=4.0.0         # Interface de démonstration
tqdm>=4.65.0          # Barres de progression
```

Installation complète :

```bash
pip install -r requirements.txt
```



## Lancer le projet

Avant de lancer localement le projet , il importe de suivre les étapes suivantes:

- Disposer d'une machine(ordinateur) avec au moins `Git` et  `Python` installé;
- Cloner localement(sur votre machine)  le repertoire de notre projet sur cette la plateforme [Github](https://github.com/);
- créer et activer un environnement virtuel Python; 
- Installez les dépendances du projet que vous retrouvez dans le [requirements.txt](./requirements.txt);
- Disposer en plus dans arborescence du projet  du repertoire `best_model`(sauvegarde de notre modèle) que vous pouvez  obtenir en éxécutant localement(si votre machine dispose  d'assez de ressources **GPU**) `python train.py` , dans le **cas contraire**, nous vous recommandons la plateforme [Google Colab](https://colab.research.google.com/) ,pour cela, nous vous fournissons en plus  notre [Notebook Colab](https://colab.research.google.com/drive/1SJz78eFJkzCT273h7R6_NZajl-4NnX1H?usp=sharing)  utilisé pour l'obtenir(le Repertoire`best_model`) et bien d'autres(les différents graphiques que vous trouvez dans les repertoires  [data-inspect](data-inspect) et [graphiques-du-model](graphiques-du-model).Mieux , vous pouvez le télécharger directement sur [notre repectoire HuggingFace](https://huggingface.co/jfo25/DL2-Devoir-3-bert-model-classification-articles-de-presse/tree/main), cela peut prendre quelques minutes , il s'agit en effet d'un fichier d'environ  **700MB**;
- lancer la commande `python demo.py`;
- Entrez four finir dans la barre d'adresse de votre navigateur [http://127.0.0.1:7860/]() pour voir le rendu de notre application.     

```bash

# 1. cloner le projet à partir de Github
     git clone https://github.com/langevin-programmer/bert-classification-inshort_news_data-1-2.csv

# 2. Créer un environnement virtuel
     python -m venv venv

# 3. Activer environnement virtuel(En étant dans le repertoire où se trouve le repertoire venv)

# Sur Window
  venv/Scripts/activate

# Sur MAC/Linux
  source venv/bin/activate
 
# 4. Installer les dépendances 
  pip install -r requirements.txt
 
```
### 1. Interface de démonstration

```bash
 # Lancer application Gradio, s'assurer avant d'avoir téléchargé le repertoire best_model 
 python demo.py
# Ouvrir → http://localhost:7860
```

### 2. Entraînement complet

```bash
python train.py
```

Lance le pipeline complet :

1. Chargement et nettoyage du CSV
2. Split stratifié 80 / 20 (préserve la distribution des classes)
3. Fine-tuning BERT sur `num_epochs` epochs avec suivi W&B
4. Sauvegarde automatique du meilleur checkpoint (selon `val_loss`)
5. Génération des courbes d'apprentissage et de la matrice de confusion


## Métriques évaluées

- **Accuracy** — taux de classification globale
- **F1-score** — équilibre précision / rappel
- **Matrice de confusion** — analyse des erreurs par classe
- **Courbes d'apprentissage** — détection overfitting / underfitting


## Compétences développées

**Techniques**
- Charger et tokenizer un dataset texte avec la librairie transformers (Hugging Face)
- Implémenter une classe Dataset PyTorch personnalisée pour le texte
- Fine-tuner un modèle BERT pré-entraîné avec une boucle d'entraînement PyTorch pure
- Construire et déployer une interface de démonstration avec Gradio
- Versionner et documenter un projet sur GitHub (travail collaboratif en binôme)

**Analytiques**
- Analyser les courbes d'apprentissage (overfitting, underfitting)
- Interpréter les métriques : accuracy, F1-score, matrice de confusion
- Comprendre l'apport du pré-entraînement (transfer learning) en NLP
- Rédiger une documentation technique claire (README.md)


## Notre binôme 
 
- **aljibreendiallo**: Alpha Oumar DIALLO 
- **langevin-programmer**: Jean-Fabrice OUFFOUE




