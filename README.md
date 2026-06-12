[lien du dataset](https://drive.google.com/file/d/1S6D_YMvB6W7yggkPG23Rm_u5FV9Z6_DZ/view?usp=drive_link)


### STRUCTURE DU PROJET 

```
.
└── bert-classification-inshort_news_data-1-2.csv
    ├── data
    │   └── inshort_news_data-1 2.csv 
    ├── dataset.py
    ├── demo.py
    ├── model.py
    ├── README.md
    ├── requirements.txt
    ├── train.py
    └── utils.py

```

### STRUCTURE A SUPPRIMER

```
bert-classification-nomdataset/

|-- data/                 <- dataset (ou script de téléchargement)

|-- dataset.py           <- TextClassificationDataset

|-- model.py             <- chargement / définition du modèle BERT

|-- train.py             <- boucles train_epoch / eval_epoch + main

|-- demo.py              <- interface Gradio

|-- utils.py             <- métriques, seed, visualisations

|-- requirements.txt

`-- README.md           <- rapport (bonus)

```

