# MODULE DL2(DEEP LEARNING 2) DEVOIR 3 ENTRE 11-16 JUIN 2026

# Fine-tuner un modèle BERT pour la classification de texte et déployer une démo interactive



## Objectif


Dans ce projet , nous allons **classifier la colonne "news_headline"+"news_article" de notre dataset  suivant la colonne  "news_category"**. Vous trouverez la source de  notre **dataset** à l'addresse suivante :
[lien du dataset](https://drive.google.com/file/d/1S6D_YMvB6W7yggkPG23Rm_u5FV9Z6_DZ/view?usp=drive_link)




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





## Métriques évaluées

- **Accuracy** — taux de classification globale
- **F1-score** — équilibre précision / rappel
- **Matrice de confusion** — analyse des erreurs par classe
- **Courbes d'apprentissage** — détection overfitting / underfitting





## Analyse du dataset 

Avant d’écrire la moindre ligne de code, nous inspectons le dataset :

-  Nombre total d'exemples et nombre de classes  
-  Distribution des classes , si déséquilibre > 2:1, justifiez votre stratégie 
-  Longueur des textes (min, max, moyenne en nombre de tokens) , pour choisir max_length
-  Afficher au moins 5 exemples de textes avec leurs labels

voir le repertoire [data-inspect](data-inspect)  pour plus de détails. 




## Outils

Nous utions Python 3.9+, PyTorch ≥2.0, transformers (Hugging Face), Gradio, scikit-learn GPU recommandé (Google Colab accepté)




## Lancer le projet

```bash

# cloner le projet à partir de Github
git clone https://github.com/langevin-programmer/bert-classification-inshort_news_data-1-2.csv

# Installer les dépendances
# Assurez-vous avant d'activer un environnement virtuel 
pip install -r requirements.txt

 
# Lancer application Gradio
python demo.py
 
```




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




