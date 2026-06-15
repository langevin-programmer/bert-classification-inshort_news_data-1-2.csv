"""
dataset.py — TextClassificationDataset pour le fine-tuning de BERT
====================================================================
Implémentation d'un Dataset PyTorch personnalisé pour la classification
de texte avec BERT. Gère la tokenization, le padding et les masques
d'attention.

"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Optional


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de texte avec BERT.

    Tokenize chaque texte à l'initialisation, gère le padding jusqu'à
    `max_length` et renvoie les masques d'attention nécessaires à BERT.

    Args:
        texts        (List[str]): Textes bruts à classer.
        labels       (List[int]): Labels entiers correspondants (0, 1, 2).
        tokenizer    (PreTrainedTokenizerBase): Tokenizer HuggingFace
                                               (ex. bert-base-multilingual-cased).
        max_length   (int): Longueur maximale de séquence (en tokens).
                           Choisir 128 pour les titres courts, 256 pour les articles.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        >>> ds = TextClassificationDataset(texts, labels, tokenizer, max_length=128)
        >>> sample = ds[0]
        >>> print(sample["input_ids"].shape)   # torch.Size([128])
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.texts      = [str(t).strip() for t in texts]
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Méthodes obligatoires de torch.utils.data.Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Retourne le nombre total d'exemples dans le dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenize et retourne le i-ème exemple sous forme de tenseurs PyTorch.

        Retourne un dict contenant :
            - input_ids      (LongTensor [max_length]) : IDs des tokens.
            - attention_mask (LongTensor [max_length]) : 1 = token réel, 0 = padding.
            - label          (LongTensor [])           : Label entier de la classe.

        Note importante :
            Le masque d'attention est INDISPENSABLE pour BERT : sans lui, le modèle
            traite les tokens de padding comme du contenu réel, dégradant les performances.
        """
        text  = self.texts[idx]
        label = int(self.labels[idx])

        # Tokenization : padding + truncation pour atteindre max_length uniformément
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",      # Remplit jusqu'à max_length avec [PAD]
            truncation=True,           # Tronque les textes plus longs
            return_attention_mask=True,# CRITIQUE : renvoyer le masque
            return_tensors="pt",       # Tenseurs PyTorch directement
        )

        return {
            # squeeze(0) : retire la dimension batch ajoutée par return_tensors="pt"
            "input_ids":      encoding["input_ids"].squeeze(0),       # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [max_length]
            "label":          torch.tensor(label, dtype=torch.long),  # scalaire
        }

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------

    def get_text(self, idx: int) -> str:
        """Retourne le texte brut à l'indice `idx` (pour le débogage)."""
        return self.texts[idx]

    def get_label(self, idx: int) -> int:
        """Retourne le label entier à l'indice `idx`."""
        return int(self.labels[idx])

    def __repr__(self) -> str:
        return (
            f"TextClassificationDataset("
            f"n_samples={len(self)}, "
            f"max_length={self.max_length}, "
            f"n_classes={len(set(self.labels))})"
        )

if __name__ == '__main__':
    print(' TextClassificationDataset défini.')