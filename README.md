#  BERT Fine-Tuning — Classification de News (InShort)

> **Dakar Institute of Technology (DIT)  Master 2 Intelligence Artificielle**  
> **Devoir N°3 Deep Learning**  
> **Réalisé par : Alpha Oumar DIALLO & Jean-Fabrice OUFFOUE** 
> Modèle : `google-bert/bert-base-multilingual-cased` | Dataset : InShort News | 7 catégories

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
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

## L'application est déployée sur huggingface et est accessible sur : https://huggingface.co/spaces/jfo25/DL2-Devoir-3-bert-classification-articles-de-presse 



## Vue d'ensemble


Dans ce projet , nous allons **classifier la colonne "news_headline"+"news_article" de notre dataset  suivant la colonne  "news_category"**. Vous trouverez la source de  notre **dataset** à l'addresse suivante :
[lien du dataset](https://drive.google.com/file/d/1S6D_YMvB6W7yggkPG23Rm_u5FV9Z6_DZ/view?usp=drive_link)


Ce projet fine-tune **BERT multilingue** pour classifier automatiquement des articles de presse en 7 catégories thématiques à partir du dataset **InShort News**.

L'approche optimale retenue consiste à concaténer le **titre** et le **corps de l'article** (`combined`) avant de les passer à BERT, ce qui maximise la richesse sémantique de l'entrée et la confiance des prédictions.

```
news_headline + ". " + news_article  →  BERT  →  Catégorie prédite
```

**Classes détectées :** `automobile` · `entertainment` · `politics` · `science` · `sports` · `technology` · `world`

---

## Résultats obtenus

> Métriques calculées sur le set de validation (20% du dataset, 964 exemples) — **meilleur checkpoint (epoch 2, val_loss minimale)**.

### Métriques globales

| Métrique              | Valeur     |
|-----------------------|-----------|
| **Accuracy**          | **93.98%** |
| **F1-macro**          | **93.65%** |
| Précision macro       | 93.19%     |
| Recall macro          | 94.15%     |
| Exemples val. totaux  | 964        |
| Exemples corrects     | 906        |

### Métriques par classe

| Classe          | Précision | Recall | F1-score | Support |
|-----------------|-----------|--------|----------|---------|
| automobile      | 90.6%     | 94.1%  | 92.3%    | 51      |
| entertainment   | 98.4%     | 95.0%  | 96.7%    | 200     |
| politics        | 97.3%     | 99.1%  | 98.2%    | 109     |
| science         | 88.0%     | 93.6%  | 90.7%    | 78      |
| sports          | 98.2%     | 98.2%  | 98.2%    | 171     |
| **technology**  | 86.8%     | 87.3%  | **87.0%**| 150     |
| world           | 93.1%     | 91.7%  | 92.4%    | 205     |

> **Classe la plus difficile : `technology`** (F1 = 87.0%) — confusions principalement avec `world` (9 cas) et `science` (5 cas), sémantiquement proches.  
> **Classe la mieux classifiée : `politics`** (F1 = 98.2%) — vocabulaire très distinctif.

### Courbes d'apprentissage

Les courbes montrent un entraînement sain sur 4 epochs :

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val F1 |
|-------|-----------|----------|-----------|---------|--------|
| 1     | 0.75     |  0.29    |  78%      |  92%    |  90%   |
| 2     | 0.23     |  0.25    |  93%      |  93%    |  92%   |
| 3     | 0.15     |  0.25    |  95%      |  94%    |  93%   |
| 4     | 0.11     |  0.22    |  97%      |  95%    |  94%   |

**Observations clés :**
- La `val_loss` reste stable de l'epoch 1 à 4 sans signe d'overfitting significatif
- La `train_loss` chute fortement (0.75 → 0.11), signe d'un bon apprentissage
- La `val_accuracy` démarre déjà à ~92% dès l'epoch 1, témoignant de la puissance des représentations pré-entraînées de BERT
- Le **meilleur checkpoint est sauvegardé à l'epoch 2** (val_loss minimale =  0.25)

### Matrice de confusion

```
                 automobile  entertainment  politics  science  sports  technology  world
automobile  [        48              0         0        0        0          3        0  ]
entertainment[        0             190         1        0        3          3        3  ]
politics    [         0              1        108        0        0          0        0  ]
science     [         0              0          0       73        0          5        0  ]
sports      [         0              1          0        0      168          0        2  ]
technology  [         5              0          0        5        0        131        9  ]
world       [         0              1          2        5        0          9      188  ]
```

**Principales confusions détectées :**

| Vrai label   | Prédit comme | Nb | Explication                                      |
|--------------|--------------|----|--------------------------------------------------|
| technology   | world        |  9 | Articles tech à portée géopolitique              |
| world        | technology   |  9 | Actualités mondiales liées à la tech             |
| world        | science      |  5 | Découvertes scientifiques à impact mondial       |
| technology   | science      |  5 | Recherche appliquée à la frontière des deux      |
| science      | technology   |  5 | Innovations scientifiques à connotation tech     |
| automobile   | technology   |  3 | Véhicules électriques / technologie automobile   |
| entertainment| sports       |  3 | E-sport ou célébrités du sport                   |

---

## Architecture du projet

```
bert_news/
├── dataset.py        # Dataset PyTorch (tokenisation, padding, masques d'attention)
├── model.py          # CONFIG centralisé + chargement/sauvegarde du modèle BERT
├── train.py          # Pipeline complet : données, entraînement, évaluation, inférence
├── demo.py           # Interface Gradio interactive
├── utils.py          # Seed, métriques, visualisations, Weights & Biases
|-- notebook.ipynb    # notebook exécuté et structuré du projet
└── requirements.txt  # Dépendances Python
```

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

---

## Installation

### Prérequis

- Python ≥ 3.9
- GPU recommandé (CUDA) - le fine-tuning CPU est possible mais prend plusieurs heures

### Étapes

```bash
# 1. Cloner / télécharger le projet
git clone <repo_url>
cd bert_news

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) Connexion Weights & Biases
wandb login
```

---

## Données

### Dataset attendu

Placer le fichier CSV dans `data/` :

```
data/
└── inshort_news_data-1 2.csv
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

---

## Utilisation

### 1. Entraînement complet

```bash
python train.py
```

Lance le pipeline complet :

1. Chargement et nettoyage du CSV
2. Split stratifié 80 / 20 (préserve la distribution des classes)
3. Fine-tuning BERT sur `num_epochs` epochs avec suivi W&B
4. Sauvegarde automatique du meilleur checkpoint (selon `val_loss`)
5. Génération des courbes d'apprentissage et de la matrice de confusion

### 2. Inspection du dataset uniquement

```bash
python train.py --inspect
```

Analyse complète du dataset : distributions, longueurs BERT par colonne, exemples. Aucun entraînement lancé.

### 3. Tests d'inférence

```bash
python train.py --test-inference
```

Charge le modèle sauvegardé dans `best_model/` et teste 3 scénarios sur un article prédéfini :

- **Cas 1 :** titre + article (cas idéal )
- **Cas 2 :** titre seul (fallback automatique )
- **Cas 3 :** article seul (fallback automatique )

Puis effectue une validation sur 10 exemples réels du dataset.

### 4. Interface de démonstration

```bash
python demo.py
# Ouvrir → http://localhost:7860
```

---

## Structure du code

### `model.py` — Configuration & modèle

Contient le dictionnaire `CONFIG` centralisé et toutes les fonctions de gestion du modèle.

```python
from model import CONFIG, load_model, load_tokenizer, save_model, load_saved_model
```

| Fonction             | Rôle                                                      |
|----------------------|-----------------------------------------------------------|
| `load_tokenizer()`   | Télécharge le tokenizer depuis HuggingFace Hub            |
| `load_model()`       | Charge BERT avec tête de classification linéaire          |
| `save_model()`       | Sauvegarde modèle + tokenizer + `class_names.json`        |
| `load_saved_model()` | Recharge un checkpoint pour l'inférence ou la démo        |

### `dataset.py` — Dataset PyTorch

```python
from dataset import TextClassificationDataset
```

Tokenise chaque texte à l'initialisation et retourne pour chaque exemple un dictionnaire de tenseurs :

```python
{
    'input_ids':      LongTensor [max_length],   # IDs des tokens BERT
    'attention_mask': LongTensor [max_length],   # 1=token réel, 0=padding
    'label':          LongTensor [],             # Indice entier de la classe
}
```

> Le masque d'attention est **indispensable** : sans lui, BERT traite les tokens de padding `[PAD]` comme du contenu réel, ce qui dégrade les représentations.

### `train.py` — Pipeline principal

Fichier central organisé en 5 sections :

| Section | Fonctions principales | Rôle |
|---------|----------------------|------|
| 1. Données | `load_and_inspect_data()`, `inspect_dataset_full()` | Chargement, nettoyage, encodage des labels |
| 2. Entraînement | `train_epoch()` | Forward + backward + gradient clipping + scheduler |
| 3. Évaluation | `eval_epoch()` | Inférence sans gradient, calcul des métriques |
| 4. Inférence | `predict()`, `print_prediction()` | Prédiction sur un article, mode combined |
| 5. Pipelines | `run_training()`, `run_test_inference()` | Orchestration complète via CLI |

**Entrées CLI :**
```bash
python train.py                   # Entraînement complet (run_training)
python train.py --inspect         # Inspection dataset (inspect_dataset_full)
python train.py --test-inference  # Tests inférence (run_test_inference)
```

### `utils.py` — Utilitaires

| Fonction                        | Rôle                                             |
|---------------------------------|--------------------------------------------------|
| `set_seed(seed)`                | Reproductibilité totale (Python, NumPy, PyTorch) |
| `compute_metrics()`             | Accuracy + F1-macro (sklearn)                    |
| `plot_confusion_matrix()`       | Heatmap seaborn, sauvegarde PNG                  |
| `plot_learning_curves()`        | Courbes loss / accuracy / F1 par epoch           |
| `plot_class_distribution()`     | Barplot de la distribution des classes           |
| `init_wandb()`                  | Initialisation d'un run Weights & Biases         |
| `log_epoch_metrics()`           | Log métriques par epoch sur W&B                  |
| `log_confusion_matrix_wandb()`  | Matrice de confusion interactive W&B             |
| `log_image_wandb()`             | Log d'une image PNG sur W&B                      |
| `log_final_summary()`           | Résumé final du run (best_val_*)                 |

---

## Hyperparamètres

Tous les hyperparamètres sont centralisés dans `CONFIG` (`model.py`) :

| Paramètre        | Valeur                                     | Description                             |
|------------------|--------------------------------------------|-----------------------------------------|
| `model_name`     | `google-bert/bert-base-multilingual-cased` | Modèle pré-entraîné HuggingFace         |
| `max_length`     | `256`                                      | Longueur max de tokenisation (tokens)   |
| `num_epochs`     | `4`                                        | Nombre d'epochs d'entraînement          |
| `batch_size`     | `16`                                       | Taille des mini-batchs                  |
| `learning_rate`  | `3e-5`                                     | Taux d'apprentissage AdamW              |
| `weight_decay`   | `0.01`                                     | Décroissance de poids (L2)              |
| `warmup_ratio`   | `0.10`                                     | 10% des steps en phase de warmup        |
| `clip_grad`      | `1.0`                                      | Gradient clipping (norme maximale)      |
| `val_split`      | `0.20`                                     | Proportion du set de validation         |
| `seed`           | `42`                                       | Graine aléatoire (reproductibilité)     |
| `save_dir`       | `best_model`                               | Répertoire de sauvegarde du checkpoint  |

### Justification des choix

**`max_length = 256`** — couvre ~95% des paires titre+article sans troncature, tout en restant bien en dessous de la limite de 512 tokens de BERT. À vérifier avec `--inspect` sur votre dataset.

**`learning_rate = 3e-5`** — plage standard pour le fine-tuning BERT (entre 2e-5 et 5e-5). Des valeurs trop élevées causent un *catastrophic forgetting* des représentations pré-entraînées.

**`warmup_ratio = 0.10`** — le learning rate monte linéairement pendant les 10 premiers % des steps avant de décroître. Cela stabilise les premières mises à jour et protège les poids pré-entraînés.

**`clip_grad = 1.0`** — évite l'explosion du gradient, particulièrement utile avec les Transformers dont les gradients peuvent être instables en début d'entraînement.

**`weight_decay = 0.01`** — régularisation L2 découplée (AdamW), réduit le risque d'overfitting sans pénaliser les biais et couches de normalisation.

---

## Approche combinée

L'entrée BERT est construite en concaténant le titre et l'article avec un point-espace :

```
Texte d'entrée = headline + ". " + article
     ↓
[CLS] titre . article [SEP]
     ↓
Tokenisation + padding → max_length = 256 tokens
     ↓
BERT encoder → [CLS] embedding → Linear → Softmax → Classe
```

### Comportement selon les champs disponibles

| Situation              | `headline` | `article` | Texte envoyé à BERT          | Qualité       |
|------------------------|------------|-----------|------------------------------|---------------|
| **Cas idéal**          | ok          | ok         | `headline + ". " + article`  | Maximale      |
| **Fallback titre**     | ok          | vide      | `headline` seul              | Réduite       |
| **Fallback article**   | vide       | ok         | `article` seul               | Réduite       |
| **Erreur**             | vide       | vide      | `ValueError` levée           | —             |

Le fallback est géré **automatiquement** dans `predict()`. Le code appelant n'a jamais besoin de tester les cas — il passe toujours `mode='combined'`.

> Les résultats obtenus (F1 = 93.65%) sont issus du cas idéal titre+article, aligné avec le mode d'entraînement. Les cas de fallback produisent des prédictions valides mais la confiance est généralement plus faible.

---

## Analyse des résultats

### Ce que montrent les courbes d'apprentissage

**Cross-Entropy Loss :**
- Train loss : chute rapide de 0.75 (epoch 1) à 0.11 (epoch 4) - le modèle apprend efficacement
- Val loss : descend de 0.29 à 0.22 de façon progressive et régulière
- L'écart train/val reste faible → **pas d'overfitting significatif**

**Accuracy :**
- Val accuracy démarre déjà à 92% dès l'epoch 1, confirmant que les représentations pré-entraînées de BERT multilingue sont directement exploitables
- La progression train (78% → 97%) vs val (92% → 95%) est normale : le modèle apprend plus vite sur train mais généralise bien

**F1-macro :**
- Courbe verte stable et croissante : 90% → 94%
- La progression entre epochs 1 et 4 est régulière, sans plateau brutal ni dégradation

### Ce que montre la matrice de confusion

La diagonale dominante confirme une très bonne classification sur l'ensemble des 7 classes. Les erreurs se concentrent sur les **frontières sémantiques naturelles** entre catégories proches :

- **technology ↔ world** (9 erreurs dans chaque sens) : articles sur la géopolitique des Big Tech, régulations numériques mondiales
- **technology ↔ science** (5+5 erreurs) : recherche fondamentale appliquée, IA, biotechnologies
- **world ↔ science** (5 erreurs) : grandes découvertes à impact mondial

Ces confusions sont **sémantiquement attendues** et difficiles à éliminer sans données supplémentaires ou augmentation ciblée sur ces classes.

### Points forts du modèle

- `politics` : F1 = 98.2% — vocabulaire politique très spécifique et distinctif
- `sports` : F1 = 98.2% — noms propres (équipes, compétitions) très discriminants
- `entertainment` : F1 = 96.7% — célébrités et culture populaire bien séparées

### Piste d'amélioration

Pour améliorer `technology` (F1 = 87%) :
1. Augmenter les données de la classe (sur-échantillonnage ou data augmentation)
2. Ajouter des exemples ambigus technology/world/science au dataset d'entraînement
3. Utiliser `class_weight` dans `CrossEntropyLoss` pour pénaliser davantage les erreurs sur `technology`

---

## Interface de démonstration

```bash
python demo.py
# Ouvrir → http://localhost:7860
```

### Fonctionnalités

- Saisie libre du titre et/ou de l'article
- Prédiction instantanée avec score de confiance
- Graphique interactif des probabilités sur toutes les classes
- Tableau détaillé avec barres de progression
- 6 exemples prédéfinis couvrant chaque domaine (espace, finance, sport, santé, tech, fallback)
- Indicateur du mode d'entrée (cas idéal ou fallback)

### Génération d'un lien public

Pour partager la démo sans configuration réseau, modifier dans `demo.py` :

```python
demo.launch(share=True)   # Génère un lien public temporaire (72h)
```

---

## Fichiers générés

Après l'exécution complète, les fichiers suivants sont créés :

| Fichier                               | Généré par          | Contenu                                           |
|---------------------------------------|---------------------|---------------------------------------------------|
| `best_model/`                         | `train.py`          | Checkpoint BERT + tokenizer + `class_names.json`  |
| `best_model/class_names.json`         | `train.py`          | Mapping indice → nom de classe                    |
| `class_distribution.png`             | `train.py`          | Distribution des classes (set d'entraînement)     |
| `class_distribution_inspection.png`  | `--inspect`         | Distribution + camembert détaillé                 |
| `token_length_distribution.png`      | `--inspect`         | Histogrammes longueurs BERT par colonne           |
| `confusion_matrix.png`               | `train.py`          | Matrice de confusion du meilleur checkpoint       |
| `learning_curves.png`                | `train.py`          | Courbes loss / accuracy / F1 par epoch            |

---

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

---

*Alpha Oumar DIALLO & Jean-Fabrice OUFFOUE*  
*Modèle : `google-bert/bert-base-multilingual-cased`*
