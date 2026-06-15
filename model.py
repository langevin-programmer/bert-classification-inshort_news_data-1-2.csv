"""
model.py :  Chargement et définition du modèle BERT pour la classification
==========================================================================
Charge `google-bert/bert-base-multilingual-cased` avec une tête de
classification linéaire (BertForSequenceClassification), sauvegarde et
recharge le meilleur checkpoint.

Note : On utilise `AutoModelForSequenceClassification` (et NON
`AutoModelForMaskedLM`) car notre tâche est la classification de séquences,
pas le remplissage de masques. HuggingFace remplace automatiquement la tête
MLM par une couche linéaire de classification.

"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "google-bert/bert-base-multilingual-cased"
DEFAULT_SAVE_DIR = "best_model"


# ──────────────────────────────────────────────────────────────────────────────
# Chargement depuis HuggingFace Hub
# ──────────────────────────────────────────────────────────────────────────────


def load_tokenizer(model_name: str = DEFAULT_MODEL) -> AutoTokenizer:
    """
    Charge le tokenizer BERT multilingue depuis HuggingFace Hub.

    Args:
        model_name (str): Identifiant HuggingFace du modèle.

    Returns:
        AutoTokenizer: Tokenizer WordPiece multilingue (119K vocab).

    Example:
        >>> tokenizer = load_tokenizer()
        >>> tokens = tokenizer("Hello world", return_tensors="pt")
    """
    print(f"    Téléchargement du tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   Tokenizer chargé | Vocab size : {tokenizer.vocab_size:,}")
    return tokenizer


def load_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL,
) -> AutoModelForSequenceClassification:
    """
    Charge BERT pré-entraîné avec une tête de classification linéaire.

    Architecture résultante :
        BertModel (encodeur) → Dropout → Linear(hidden_size, num_classes)

    La tête linéaire est initialisée aléatoirement ; les poids de l'encodeur
    sont issus du pré-entraînement et seront affinés (fine-tuned).

    Args:
        num_classes (int): Nombre de classes cibles 
        model_name  (str): Identifiant HuggingFace du modèle.

    Returns:
        AutoModelForSequenceClassification: Modèle BERT prêt pour le fine-tuning.

    Note:
        `ignore_mismatched_sizes=True` est nécessaire car la tête de
        classification remplace la tête MLM du modèle pré-entraîné.
    """
    print(f"  Chargement du modèle BERT : {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Remplace la tête MLM → classification
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Modèle chargé")
    print(f"  Paramètres totaux     : {total_params:,}")
    print(f"  Paramètres entraînables : {trainable_params:,}")
    print(f"  Nombre de classes     : {num_classes}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ──────────────────────────────────────────────────────────────────────────────


def save_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    class_names: list,
    save_dir: str = DEFAULT_SAVE_DIR,
) -> None:
    """
    Sauvegarde le modèle, le tokenizer et le mapping des classes.

    Crée `save_dir/` avec :
        - config.json, pytorch_model.bin (ou model.safetensors)
        - tokenizer_config.json, vocab.txt, special_tokens_map.json
        - class_names.json  ← mapping index → nom de classe

    Args:
        model       : Modèle fine-tuné.
        tokenizer   : Tokenizer associé.
        class_names (list): Liste ordonnée des noms de classes.
        save_dir    (str) : Chemin du répertoire de destination.
    """
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Sauvegarde du mapping des classes (nécessaire pour la démo Gradio)
    class_path = os.path.join(save_dir, "class_names.json")
    with open(class_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"   Checkpoint sauvegardé → {save_dir}/")
    print(f"     Classes : {class_names}")


# ──────────────────────────────────────────────────────────────────────────────
# Rechargement
# ──────────────────────────────────────────────────────────────────────────────


def load_saved_model(
    save_dir: str = DEFAULT_SAVE_DIR,
) -> tuple:
    """
    Recharge un modèle fine-tuné et son tokenizer depuis un répertoire.

    Utilisé principalement par `demo.py` pour l'inférence Gradio.

    Args:
        save_dir (str): Chemin contenant les fichiers sauvegardés.

    Returns:
        tuple: (model, tokenizer, class_names)
            - model      : AutoModelForSequenceClassification en mode eval.
            - tokenizer  : AutoTokenizer associé.
            - class_names: Liste des noms de classes.

    Raises:
        FileNotFoundError: Si `save_dir` n'existe pas ou est incomplet.
    """
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(
            f"Le répertoire '{save_dir}' est introuvable. "
            "Veuillez d'abord entraîner le modèle avec : python train.py"
        )

    tokenizer   = AutoTokenizer.from_pretrained(save_dir)
    model       = AutoModelForSequenceClassification.from_pretrained(save_dir)

    class_path  = os.path.join(save_dir, "class_names.json")
    with open(class_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model.eval()  # Désactive dropout pour l'inférence

    print(f"   Modèle rechargé depuis '{save_dir}/'")
    print(f"     Classes ({len(class_names)}) : {class_names}")

    return model, tokenizer, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Test rapide
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n Test du module model.py\n")
    tok   = load_tokenizer()
    model = load_model(num_classes=7)
    print("\n Architecture de la tête de classification :")
    print(model.classifier)
