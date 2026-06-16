import random
import numpy as np
import torch
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import wandb



# Hyperparamètres (CONFIG)
CONFIG = {
    # Modèle
    'model_name':     'google-bert/bert-base-multilingual-cased',
    # max_length : 128 pour titres seuls, 256 pour titre+article concaténés
    # (BERT accepte jusqu'à 512 tokens, mais 256 couvre ~95% des paires)
    'max_length':     256,

    # Données
    'dataset_path':   'data/inshort_news_data-1-2.csv',  #  Adaptez ce chemin
    'text_column':    'news_headline',    # Colonne titre
    'article_column': 'news_article',     # Colonne article (concaténé au titre)
    'label_column':   'news_category',
    'val_split':      0.20,

    # Entraînement
    'num_epochs':    4,
    'batch_size':    16,
    'learning_rate': 3e-5,
    'weight_decay':  0.01,
    'warmup_ratio':  0.10,
    'clip_grad':     1.0,

    # Reproductibilité & sauvegarde
    'seed':          42,
    'save_dir':      'best_model',

    # W&B
    'project_wandb': 'bert-classification-inshort-news',
    'run_name':      'bert-multilingual-cased-run1',
}


# 1- Reproductibilité

def set_seed(seed: int = 42) -> None:
    """
    Fixe toutes les sources d'aléatoire (Python, NumPy, PyTorch, cuDNN)
    pour garantir la reproductibilité des expériences.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f'🌱 Seed fixée à {seed} — résultats reproductibles.')


# 2- Métriques

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Calcule accuracy et F1-score macro.

    Note : F1-macro donne le même poids à chaque classe, indépendamment
    de sa fréquence — métrique appropriée pour les datasets déséquilibrés.
    """
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if verbose and class_names:
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
        print(f'\n Rapport de classification :\n{report}')
    return {'accuracy': float(acc), 'f1_macro': float(f1_macro)}


# 3- Visualisations

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str = 'confusion_matrix.png',
    title: str = 'Matrice de Confusion — BERT Fine-tuning',
) -> str:
    """Génère et sauvegarde la matrice de confusion sous forme de heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.2),
                                    max(6, len(class_names))))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='white', ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Vrai Label', fontsize=11)
    ax.set_xlabel('Label Prédit', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'  Matrice de confusion → {save_path}')
    return os.path.abspath(save_path)


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: str = 'learning_curves.png',
) -> str:
    """Trace les courbes de loss et d'accuracy par epoch."""
    epochs = range(1, len(history['train_loss']) + 1)
    has_f1 = 'val_f1' in history and len(history['val_f1']) > 0
    n_plots = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle(
        'Courbes d\'apprentissage — BERT Fine-tuning (InShort News)',
        fontsize=13, fontweight='bold', y=1.02
    )
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train', linewidth=2, markersize=5)
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val',   linewidth=2, markersize=5)
    axes[0].set_title('Cross-Entropy Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['train_accuracy'], 'b-o', label='Train', linewidth=2, markersize=5)
    axes[1].plot(epochs, history['val_accuracy'],   'r-o', label='Val',   linewidth=2, markersize=5)
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    if has_f1:
        axes[2].plot(epochs, history['val_f1'], 'g-o', label='Val F1-macro', linewidth=2, markersize=5)
        axes[2].set_title('F1-Score Macro (val)', fontweight='bold')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('F1-macro')
        axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'   Courbes d\'apprentissage → {save_path}')
    return os.path.abspath(save_path)


def plot_class_distribution(
    labels: List[int],
    class_names: List[str],
    title: str = 'Distribution des classes',
    save_path: str = 'class_distribution.png',
) -> str:
    """Trace la distribution des classes dans le dataset."""
    counts = [labels.count(i) for i in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, counts, color=plt.cm.Set3.colors[:len(class_names)])
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Classe'); ax.set_ylabel('Nombre d\'exemples')
    ax.tick_params(axis='x', rotation=30)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'  Distribution des classes → {save_path}')
    return os.path.abspath(save_path)


# 4- Weights & Biases

def init_wandb(project_name: str, config: dict, run_name: str = None):
    """Initialise un run Weights & Biases."""
    run = wandb.init(
        project=project_name, name=run_name, config=config,
        tags=['bert', 'nlp', 'classification', 'fine-tuning', 'master2', 'DIT'],
    )
    print(f'   W&B initialisé → {run.url}')
    return run


def log_epoch_metrics(
    epoch, train_loss, val_loss, train_acc, val_acc, val_f1, lr
) -> None:
    """Logue les métriques d'une epoch sur W&B."""
    wandb.log(
        {'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss,
         'train/accuracy': train_acc, 'val/accuracy': val_acc,
         'val/f1_macro': val_f1, 'train/lr': lr},
        step=epoch,
    )


def log_confusion_matrix_wandb(y_true, y_pred, class_names) -> None:
    """Logue la matrice de confusion interactive sur W&B."""
    wandb.log({'val/confusion_matrix': wandb.plot.confusion_matrix(
        probs=None, y_true=y_true, preds=y_pred,
        class_names=class_names, title='Matrice de Confusion (validation)',
    )})


def log_image_wandb(key: str, image_path: str, caption: str = '') -> None:
    """Logue une image sur W&B."""
    wandb.log({key: wandb.Image(image_path, caption=caption)})


def log_final_summary(
    best_val_loss, best_val_acc, best_val_f1, class_names
) -> None:
    """Logue le résumé final du run sur W&B."""
    wandb.run.summary.update({
        'best/val_loss': best_val_loss, 'best/val_accuracy': best_val_acc,
        'best/val_f1_macro': best_val_f1, 'n_classes': len(class_names),
        'classes': ', '.join(class_names),
    })
    print(
        f'\n   Résumé W&B enregistré :\n'
        f'     best_val_loss = {best_val_loss:.4f}\n'
        f'     best_val_acc  = {best_val_acc:.4f}\n'
        f'     best_val_f1   = {best_val_f1:.4f}'
    )


print(' Fonctions utils.py définies.')

