import gradio as gr

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import matplotlib.pyplot as plt

# 1. CHARGER LE MODÈLE AU DÉMARRAGE (local)
# best_model est le repertoire contenant la sauvegarde de notre modèle
# Vous pouvez le retrouver(repertoire best_model) à l'addresse suivante :
# https://huggingface.co/jfo25/DL2-Devoir-3-bert-model-classification-articles-de-presse
# précisement dans Files and versions
model = AutoModelForSequenceClassification.from_pretrained("best_model")
tokenizer = AutoTokenizer.from_pretrained("best_model")
class_names = json.load(open("best_model/class_names.json"))

model.eval() # mode prédiction (pas entraînement)

# 2. FONCTION DE CLASSIFICATION
def classify_news(headline: str, article: str):

    # Fallback automatique
    if headline.strip() and article.strip():
        text = headline + " " + article
        mode = "combined (titre + article)"
    elif headline.strip():
        text = headline
        mode = "headline uniquement"
    elif article.strip():
        text = article
        mode = "article uniquement"
    else:
        return " Veuillez saisir un titre ou un article.", "", None

    # Tokenizer
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    scores = {
        class_names[i]: float(probs[0][i])
        for i in range(len(class_names))
    }

    # Catégorie gagnante
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    confidence_text = f"Confiance : {best_score:.1%} | Mode : {mode}"

    # Graphique
    fig, ax = plt.subplots(figsize=(6, 3))
    categories = list(scores.keys())
    values = list(scores.values())
    colors = ["#FF7C00" if c == best_label else "#AAAAAA" for c in categories]
    bars = ax.barh(categories, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score de confiance")
    ax.set_title("Distribution des scores (toutes les classes)")
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)
    plt.tight_layout()

    return f" {best_label.upper()}", confidence_text, fig

# Fonction effacer
def clear_fields():
    return "", "", "", "", None


#  INTERFACE GRADIO
with gr.Blocks() as app:
   

    gr.HTML("""
        <div class="header-box">
            <h1 style="margin:0; font-size:26px;"> Devoir n°3 Deep Learning 2: BERT News Classifier</h1>
            <p style="margin:4px 0 8px 0; color:#555; font-size:15px;">
                Classification d'articles de presse — InShort News Data
            </p>
            <span class="badge">google-bert/bert-base-multilingual-cased</span>
            <span class="badge">Mode optimal : combined</span>
            <span class="badge">Réalisé par : Alpha Oumar DIALLO & Jean-Fabrice OUFFOUE</span>
        </div>
        """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Saisie de l'article")
            headline_input = gr.Textbox(
                label="Titre (news_headline)",
                placeholder="Ex : India launches new space mission to explore the Moon",
                lines=2,
            )
            article_input = gr.Textbox(
                label="Article (news_article)",
                placeholder="Ex : The Indian Space Research Organisation (ISRO) successfully launched...",
                lines=6,
            )
            gr.Markdown(
                "> **Approche optimale :** Le modèle utilise toujours `combined` "
                "(titre + article). Si un champ est absent, le fallback s'active automatiquement.",
            )
            with gr.Row():
                clear_btn  = gr.Button("Effacer",   variant="secondary", scale=1)
                submit_btn = gr.Button("Classifier", variant="primary",   scale=2)

        with gr.Column(scale=1):
            gr.Markdown("### Résultats")
            label_output = gr.Textbox(
                label=" Catégorie prédite",
                interactive=False,
                elem_classes=["predicted-label"],
            )
            confidence_output = gr.Textbox(
                label=" Confiance & mode d'entrée",
                interactive=False,
                lines=2,
            )
            chart_output = gr.Plot(label="Distribution des scores")

            
    
    gr.HTML("""
             <div style="text-align:center; margin-top:20px; color:#888; font-size:13px;">
                    Modèle : <b>google-bert/bert-base-multilingual-cased</b> fine-tuné sur InShort News &nbsp;|&nbsp;
                    Devoir n°3 Deep Learning 2 -Dakar Institute of Technology (DIT) - Master 2 Intelligence Artificielle
                </div>
            """)
    
    submit_btn.click(
        fn=classify_news,
        inputs=[headline_input, article_input],
        outputs=[label_output, confidence_output, chart_output]
    )

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[headline_input, article_input, label_output, confidence_output, chart_output]
    )
        

    
app.launch()




    