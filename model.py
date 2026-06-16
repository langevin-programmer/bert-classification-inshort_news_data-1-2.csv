# model.py
import os 
import json


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

DEFAULT_MODEL    = 'google-bert/bert-base-multilingual-cased'
DEFAULT_SAVE_DIR = 'best_model'


def load_tokenizer(model_name: str = DEFAULT_MODEL) -> AutoTokenizer:
    """
    Charge le tokenizer BERT multilingue depuis HuggingFace Hub.
    """
    print(f'      Téléchargement du tokenizer : {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f'    Tokenizer chargé | Vocab size : {tokenizer.vocab_size:,}')
    return tokenizer


def load_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL,
) -> AutoModelForSequenceClassification:
    """
    Charge BERT pré-entraîné avec une tête de classification linéaire.

    Architecture : BertModel (encodeur) → Dropout → Linear(hidden_size, num_classes)

    Note: `ignore_mismatched_sizes=True` est nécessaire car la tête de
    classification remplace la tête MLM du modèle pré-entraîné.
    """
    print(f'   Chargement du modèle BERT : {model_name}')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Remplace la tête MLM → classification
    )
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'     Modèle chargé')
    print(f'     Paramètres totaux       : {total_params:,}')
    print(f'     Paramètres entraînables : {trainable_params:,}')
    print(f'     Nombre de classes       : {num_classes}')
    return model


def save_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    class_names: list,
    save_dir: str = DEFAULT_SAVE_DIR,
) -> None:
    """
    Sauvegarde le modèle, le tokenizer et le mapping des classes.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    class_path = os.path.join(save_dir, 'class_names.json')
    with open(class_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f'    Checkpoint sauvegardé → {save_dir}/')
    print(f'     Classes : {class_names}')


def load_saved_model(save_dir: str = DEFAULT_SAVE_DIR) -> tuple:
    """
    Recharge un modèle fine-tuné et son tokenizer depuis un répertoire.
    """
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(
            f"Le répertoire '{save_dir}' est introuvable. "
            'Veuillez d\'abord entraîner le modèle.'
        )
    tokenizer   = AutoTokenizer.from_pretrained(save_dir)
    model       = AutoModelForSequenceClassification.from_pretrained(save_dir)
    class_path  = os.path.join(save_dir, 'class_names.json')
    with open(class_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    model.eval()
    print(f"    Modèle rechargé depuis '{save_dir}/'")
    print(f'     Classes ({len(class_names)}) : {class_names}')
    return model, tokenizer, class_names


print(' Fonctions model.py définies.')