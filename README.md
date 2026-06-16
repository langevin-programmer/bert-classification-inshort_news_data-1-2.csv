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

Nous utilisons Python 3.9+, PyTorch ≥2.0, transformers (Hugging Face), Gradio, scikit-learn GPU recommandé (Google Colab accepté)

##  Déploiement de nos applications et modèles

Nous avons déployé notre application et modèle sur la plateforme [huggingface](https://huggingface.co/). 
Vous les  touverez via les différents liens ci-dessous:
- [Notre application de classification du titre + contenu d'articles de presse](https://huggingface.co/spaces/jfo25/DL2-Devoir-3-bert-classification-articles-de-presse)
- [Sauvegarde du modèle associé à l'application ci-dessus](https://huggingface.co/jfo25/DL2-Devoir-3-bert-model-classification-articles-de-presse/tree/main)


Nous vous informons par ailleurs qu'avant de déployer application ci-dessus, nous avons déployé une version qui permet de classifier uniquement les **titres** des articles(le modèle associé est en en effet entrainé uniquement sur les colonnes `news_headline` et `news_category`). Nous vous donnons également accès à cette dernière [application](https://huggingface.co/spaces/jfo25/classification-titres-articles-de-presse) ainsi qu'au [modèle](https://huggingface.co/jfo25/model-classification-titres-articles-de-presse/tree/main) associé. 




## Lancer le projet

Avant de lancer localement le projet , il importe de suivre les étapes suivantes:

- Disposer d'une machine(ordinateur) avec au moins `Git` et  `Python` installé;
- Cloner localement(sur votre machine)  le repertoire de notre projet sur cette la plateforme [Github](https://github.com/);
- créer un environnement virtuel Python; 
- Installez les dépendances du projet que vous retrouvez dans le [requirements.txt](./requirements.txt);
- Disposer en plus dans arborescence du projet  du repertoire `best_model`(sauvegarde de notre modèle) que vous pouvez  obtenir en éxécutant localement(si votre machine dispose  d'assez de ressources **GPU**) `python train.py` , dans le **cas contraire**, nous vous recommandons la plateforme [Google Colab](https://colab.research.google.com/) ,pour cela, nous vous fournissons en plus  notre [Notebook Colab](https://colab.research.google.com/drive/1SJz78eFJkzCT273h7R6_NZajl-4NnX1H?usp=sharing)  utilisé pour l'obtenir(le Repertoire`best_model`) et bien d'autres(les différents graphiques que vous trouvez dans les repertoires  [data-inspect](data-inspect) et [graphiques-du-model](graphiques-du-model).Mieux , vous pouvez le télécharger directement sur [notre repectoire HuggingFace](https://huggingface.co/jfo25/DL2-Devoir-3-bert-model-classification-articles-de-presse/tree/main), cela peut prendre quelques minutes , il s'agit en effet d'un fichier d'environ  **700MB**;
- lancer la commande `python demo.py`;
- Entrez four finir dans la barre d'adresse de votre navigateur [http://127.0.0.1:7860/]() pour voir le rendu de notre application.     

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




