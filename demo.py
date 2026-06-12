# demo.py


# Gradio est une bibliothèque pour créer des interfaces utilisateur interactives
import gradio as gr


    
#
# Fonction pour prédire si un message est du spam ou non
def bert_classication(message):
    pass
    #return

# Créer une interface Gradio pour l'application
interface_gradio = gr.Interface(fn=bert_classication, inputs="text", outputs="text",
                                title="Bert-Classification",
                                 description="Classification de texte"
                                 )
# To create a public link, set `share=True` in `launch()`.
# This share link is temporary and will last for up to 1 week (best effort). 
# For free permanent hosting and GPU upgrades, run `gradio deploy` from the 
# terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
interface_gradio.launch(share=True)