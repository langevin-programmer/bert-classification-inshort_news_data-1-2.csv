### STRUCTURE DU PROJET 

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

[lien du dataset](https://drive.google.com/file/d/1S6D_YMvB6W7yggkPG23Rm_u5FV9Z6_DZ/view?usp=drive_link)