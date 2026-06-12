## Fine-tuner un modèle BERT pour la classification de texte et déployer une démo interactive

Dans ce projet , nous allons **classifier la colonne "news_headline"+"news_article"  suivant la colonne  "news_category"**. Vous trouverez la source de  notre **dataset** à l'addresse suivant : 

[lien du dataset](https://drive.google.com/file/d/1S6D_YMvB6W7yggkPG23Rm_u5FV9Z6_DZ/view?usp=drive_link)




### STRUCTURE DU PROJET 

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




