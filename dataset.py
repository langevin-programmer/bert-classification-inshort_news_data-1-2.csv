# dataset.py 
from torch.utils.data import Dataset
from typing import  List
from transformers import PreTrainedTokenizerBase
import torch


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de texte avec BERT.

    Tokenize chaque texte à l'initialisation, gère le padding jusqu'à
    `max_length` et renvoie les masques d'attention nécessaires à BERT.

    Args:
        texts        (List[str]): Textes bruts à classer.
        labels       (List[int]): Labels entiers correspondants (0, 1, 2 …).
        tokenizer    (PreTrainedTokenizerBase): Tokenizer HuggingFace.
        max_length   (int): Longueur maximale de séquence (en tokens).
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

    def __len__(self) -> int:
        """Retourne le nombre total d'exemples dans le dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenize et retourne le i-ème exemple sous forme de tenseurs PyTorch.

        Retourne un dict contenant :
            - input_ids      (LongTensor [max_length])
            - attention_mask (LongTensor [max_length]) : 1=token réel, 0=padding
            - label          (LongTensor [])           : Label entier de la classe

        Note : Le masque d'attention est INDISPENSABLE pour BERT — sans lui,
        le modèle traite les tokens de padding comme du contenu réel.
        """
        text  = self.texts[idx]
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',       # Remplit jusqu'à max_length avec [PAD]
            truncation=True,            # Tronque les textes plus longs
            return_attention_mask=True, # CRITIQUE : renvoyer le masque
            return_tensors='pt',        # Tenseurs PyTorch directement
        )

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),       # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [max_length]
            'label':          torch.tensor(label, dtype=torch.long),  # scalaire
        }

    def get_text(self, idx: int) -> str:
        """Retourne le texte brut à l'indice `idx` (pour le débogage)."""
        return self.texts[idx]

    def get_label(self, idx: int) -> int:
        """Retourne le label entier à l'indice `idx`."""
        return int(self.labels[idx])

    def __repr__(self) -> str:
        return (
            f'TextClassificationDataset('
            f'n_samples={len(self)}, '
            f'max_length={self.max_length}, '
            f'n_classes={len(set(self.labels))})'
        )

print('  TextClassificationDataset défini.')