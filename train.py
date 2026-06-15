"""
train.py — Boucle d'entraînement PyTorch pure pour BERT (sans HF Trainer)
==========================================================================
Pipeline complet :
  1. Chargement & inspection du dataset InShort News
  2. Split train/val stratifié (80/20)
  3. Fine-tuning de bert-base-multilingual-cased avec AdamW + scheduler linéaire
  4. Sauvegarde du meilleur checkpoint (critère : val_loss minimale)
  5. Logging de toutes les métriques sur Weights & Biases

"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import wandb

from dataset import TextClassificationDataset
from model  import load_tokenizer, load_model, save_model
from utils  import (
    set_seed,
    compute_metrics,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_class_distribution,
    init_wandb,
    log_epoch_metrics,
    log_confusion_matrix_wandb,
    log_image_wandb,
    log_final_summary,
)


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMÈTRES — modifiez cette section pour vos expériences
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Modèle
    "model_name":    "google-bert/bert-base-multilingual-cased",
    "max_length":    128,          # 128 tokens suffisent pour des titres courts

    # Données
    "dataset_path":  "data/inshort_news_data-1 2.csv",
    "text_column":   "news_headline",   # Colonne texte principale
    "label_column":  "news_category",   # Colonne cible
    "val_split":     0.20,              # 20 % pour la validation

    # Entraînement
    "num_epochs":    5,
    "batch_size":    16,
    "learning_rate": 3e-5,             # Typique pour le fine-tuning BERT (2e-5 à 5e-5)
    "weight_decay":  0.01,             # L2 regularisation via AdamW
    "warmup_ratio":  0.10,             # 10 % des steps pour le warmup
    "clip_grad":     1.0,              # Gradient clipping (évite l'explosion du gradient)

    # Reproductibilité & sauvegarde
    "seed":          42,
    "save_dir":      "best_model",

    # W&B
    "project_wandb": "bert-classification-inshort-news",
    "run_name":      "bert-multilingual-cased-run1",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chargement et inspection des données
# ══════════════════════════════════════════════════════════════════════════════


def load_and_inspect_data(
    data_path: str,
    text_col: str,
    label_col: str,
) -> tuple:
    """
    Charge le CSV, inspecte la distribution des classes et affiche des exemples.

    Args:
        data_path (str): Chemin vers le fichier CSV.
        text_col  (str): Nom de la colonne texte.
        label_col (str): Nom de la colonne label.

    Returns:
        tuple: (texts, labels_int, label_encoder, class_names)
            - texts       : Liste de chaînes de caractères.
            - labels_int  : Array NumPy d'entiers (0 à N-1).
            - label_encoder: Objet LabelEncoder sklearn (pour décoder).
            - class_names : Liste des noms de classes dans l'ordre des indices.
    """
    print(f"\n{'═'*60}")
    print(f" CHARGEMENT DU DATASET")
    print(f"{'═'*60}")

    df = pd.read_csv(data_path)
    print(f"  Fichier       : {data_path}")
    print(f"  Shape brut    : {df.shape}")
    print(f"  Colonnes      : {list(df.columns)}")

    # Sélectionner et nettoyer
    df = df[[text_col, label_col]].dropna()
    df[text_col]  = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    print(f"  Après nettoyage : {len(df)} exemples")

    # Distribution des classes
    print(f"\n  Distribution des classes :")
    dist = df[label_col].value_counts()
    for cls, cnt in dist.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls:<20} {cnt:>5} ({pct:5.1f}%) {bar}")

    # Déséquilibre > 2:1 ?
    ratio = dist.max() / dist.min()
    if ratio > 2.0:
        print(f"\n   Déséquilibre détecté (ratio max/min = {ratio:.1f}x)")
        print(f"     → Stratégie : split stratifié + F1-macro comme métrique principale")

    # Longueur des textes (en tokens approx. = mots)
    lengths = df[text_col].str.split().str.len()
    print(f"\n   Longueur des textes (mots) :")
    print(f"    Min    : {lengths.min()}")
    print(f"    Max    : {lengths.max()}")
    print(f"    Moyenne: {lengths.mean():.1f}")
    print(f"    P95    : {lengths.quantile(0.95):.0f}  ← choisir max_length ≥ cette valeur")

    # 5 exemples
    print(f"\n   5 exemples de textes :")
    for i in range(min(5, len(df))):
        print(f"    [{df[label_col].iloc[i]:<12}] {df[text_col].iloc[i][:90]}...")

    # Encodage des labels (str → int)
    le = LabelEncoder()
    labels_int  = le.fit_transform(df[label_col].values)
    class_names = list(le.classes_)
    texts       = df[text_col].values.tolist()

    print(f"\n    Mapping label → indice :")
    for idx, name in enumerate(class_names):
        print(f"    {idx} → {name}")

    return texts, labels_int, le, class_names


# ══════════════════════════════════════════════════════════════════════════════
# 2. Boucle d'entraînement (une epoch)
# ══════════════════════════════════════════════════════════════════════════════


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epoch_num: int,
) -> tuple:
    """
    Effectue une epoch complète d'entraînement.

    Procédure par batch :
        1. Transfert des tenseurs sur `device`.
        2. Zero_grad pour effacer les gradients accumulés.
        3. Forward pass (logits via BERT).
        4. Calcul de la loss (CrossEntropyLoss, attend des entiers).
        5. Backward pass (calcul des gradients).
        6. Gradient clipping (évite l'explosion du gradient).
        7. Mise à jour des paramètres (optimizer.step).
        8. Mise à jour du scheduler.

    Args:
        model     : Modèle BERT en mode .train().
        loader    : DataLoader d'entraînement.
        optimizer : AdamW.
        scheduler : Scheduler linéaire avec warmup.
        criterion : CrossEntropyLoss.
        device    : CPU ou CUDA.
        epoch_num : Numéro d'epoch (pour la barre de progression).

    Returns:
        tuple: (avg_loss, accuracy, all_preds, all_labels)
    """
    model.train()  # Active dropout et batch normalisation
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    pbar = tqdm(
        loader,
        desc=f"   Epoch {epoch_num} [Train]",
        leave=False,
        unit="batch",
    )

    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        # Forward 
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # NB : on ne passe PAS labels ici pour garder le contrôle sur la loss
        )
        logits = outputs.logits          # [batch_size, num_classes]

        #  Loss 
        # CrossEntropyLoss attend : logits [N, C] et labels entiers [N]
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Backward 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=CONFIG["clip_grad"]
        )
        optimizer.step()
        scheduler.step()

        # Prédictions 
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())

        # Mise à jour de la barre et log W&B par batch
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        wandb.log({"batch/train_loss": loss.item(), "batch/step": step})

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics["accuracy"], all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# 3. Boucle d'évaluation (une epoch)
# ══════════════════════════════════════════════════════════════════════════════


def eval_epoch(
    model,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    epoch_num: int,
) -> tuple:
    """
    Évalue le modèle sur un DataLoader sans mise à jour des gradients.

     Points critiques :
        - model.eval() : désactive dropout et batch normalisation.
        - torch.no_grad() : désactive le graphe de calcul → 3-4× plus rapide,
          moins de mémoire.
        - La loss attend des labels ENTIERS (pas de one-hot encoding).

    Args:
        model     : Modèle BERT en mode .eval().
        loader    : DataLoader de validation ou de test.
        criterion : CrossEntropyLoss.
        device    : CPU ou CUDA.
        epoch_num : Numéro d'epoch (pour la barre de progression).

    Returns:
        tuple: (avg_loss, accuracy, f1_macro, all_preds, all_labels)
    """
    model.eval()  # ← Désactive dropout pour des prédictions stables
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    pbar = tqdm(
        loader,
        desc=f"   Epoch {epoch_num} [Val  ]",
        leave=False,
        unit="batch",
    )

    with torch.no_grad():   # ici pas de calcul de gradients pendant la validation
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics["accuracy"], metrics["f1_macro"], all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# 4. Fonction principale
# ══════════════════════════════════════════════════════════════════════════════


def main():
    """
    Orchestration complète de l'entraînement :
        Données → Tokenization → Split → Entraînement → Évaluation → Sauvegarde
    """

    t_start = time.time()
    print(f"\n{'═'*60}")
    print(f"   BERT FINE-TUNING — InShort News Classification")
    print(f"   Dakar Institute of Technology (DIT) — Master 2 IA")
    print(f"{'═'*60}")

    # 1. Seed & device 
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  🖥️  Device : {device}")
    if device.type == "cuda":
        print(f"     GPU : {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 2. W&B 
    run = init_wandb(CONFIG["project_wandb"], CONFIG, CONFIG["run_name"])

    # 3. Données 
    texts, labels, label_encoder, class_names = load_and_inspect_data(
        CONFIG['dataset_path'],
        CONFIG['text_column'],
        CONFIG['article_column'],
        CONFIG['label_column'],
    )
    CONFIG["num_classes"] = len(class_names)
    CONFIG["class_names"] = class_names
    wandb.config.update({"num_classes": len(class_names), "class_names": class_names})

    # Graphique distribution des classes → W&B
    dist_path = plot_class_distribution(
        list(labels), class_names, save_path="class_distribution.png"
    )
    log_image_wandb("data/class_distribution", dist_path, "Distribution des classes")

    # 4. Split stratifié 80/20 
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=CONFIG["val_split"],
        random_state=CONFIG["seed"],
        stratify=labels,             # Stratification pour préserver la distribution
    )
    print(f"\n  Split : {len(X_train)} train | {len(X_val)} val")

    # 5. Tokenizer & Datasets 
    print(f"\n   Chargement du tokenizer…")
    tokenizer = load_tokenizer(CONFIG["model_name"])

    train_dataset = TextClassificationDataset(
        X_train, y_train, tokenizer, CONFIG["max_length"]
    )
    val_dataset = TextClassificationDataset(
        X_val, y_val, tokenizer, CONFIG["max_length"]
    )

    # num_workers=0 pour compatibilité Colab
    nw = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )

    #  6. Modèle 
    print(f"\n   Chargement du modèle BERT…")
    model = load_model(CONFIG["num_classes"], CONFIG["model_name"])
    model = model.to(device)

    # Sauvegarder le mapping des classes dès maintenant
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    with open(os.path.join(CONFIG["save_dir"], "class_names.json"), "w") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # 7. Optimiseur & Scheduler 
    # AdamW : Adam avec décroissance de poids découplée (meilleur pour BERT)
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        eps=1e-8,
    )

    total_steps  = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    # Scheduler linéaire : lr monte pendant warmup_steps, puis décroît linéairement
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n  Optimiseur : AdamW | lr={CONFIG['learning_rate']} | wd={CONFIG['weight_decay']}")
    print(f"     Total steps : {total_steps} | Warmup steps : {warmup_steps}")

    # ── 8. Fonction de loss ───────────────────────────────────────────────────
    # CrossEntropyLoss attend des logits (pas de softmax avant) et des labels entiers
    criterion = nn.CrossEntropyLoss()

    # ── 9. Boucle d'entraînement ──────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "val_f1": [],
    }
    best_val_loss    = float("inf")
    best_val_acc     = 0.0
    best_val_f1      = 0.0
    best_val_preds   = []
    best_val_labels  = []

    print(f"\n{'═'*60}")
    print(f"   DÉMARRAGE DE L'ENTRAÎNEMENT ({CONFIG['num_epochs']} epochs)")
    print(f"{'═'*60}\n")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n  ┌─ Epoch {epoch}/{CONFIG['num_epochs']} {'─'*40}")

        # Entraînement
        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        # Validation
        val_loss, val_acc, val_f1, val_preds, val_labels_list = eval_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        current_lr = scheduler.get_last_lr()[0]

        # Affichage
        print(f"  │  Train : loss={train_loss:.4f} | acc={train_acc:.4f}")
        print(f"  │  Val   : loss={val_loss:.4f} | acc={val_acc:.4f} | f1={val_f1:.4f}")
        print(f"  │  LR    : {current_lr:.2e}")

        # Logging W&B
        log_epoch_metrics(
            epoch, train_loss, val_loss, train_acc, val_acc, val_f1, current_lr
        )

        # Sauvegarde du meilleur modèle (critère : val_loss minimale)
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_val_acc    = val_acc
            best_val_f1     = val_f1
            best_val_preds  = val_preds
            best_val_labels = val_labels_list

            save_model(model, tokenizer, class_names, CONFIG["save_dir"])
            print(f"  │  Meilleur modèle sauvegardé ! (val_loss={val_loss:.4f})")
        else:
            print(f"  │  (Pas d'amélioration. Meilleur val_loss: {best_val_loss:.4f})")

        print(f"  └{'─'*50}")

    # ── 10. Visualisations finales ────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"   GÉNÉRATION DES VISUALISATIONS")
    print(f"{'═'*60}")

    # Matrice de confusion (sur les prédictions du meilleur modèle)
    cm_path = plot_confusion_matrix(best_val_labels, best_val_preds, class_names)
    log_image_wandb("val/confusion_matrix_img", cm_path, "Matrice de confusion (meilleur modèle)")
    log_confusion_matrix_wandb(best_val_labels, best_val_preds, class_names)

    # Courbes d'apprentissage
    lc_path = plot_learning_curves(history)
    log_image_wandb("train/learning_curves", lc_path, "Courbes d'apprentissage")

    # Résumé W&B
    log_final_summary(best_val_loss, best_val_acc, best_val_f1, class_names)

    # Rapport de classification final
    print(f"\n   Rapport de classification final (meilleur checkpoint) :")
    compute_metrics(best_val_labels, best_val_preds, class_names, verbose=True)

    wandb.finish()

    # ── Résumé terminal ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"   ENTRAÎNEMENT TERMINÉ en {elapsed/60:.1f} min")
    print(f"  Meilleur checkpoint : '{CONFIG['save_dir']}/'")
    print(f"  best_val_loss  = {best_val_loss:.4f}")
    print(f"  best_val_acc   = {best_val_acc:.4f}")
    print(f"  best_val_f1    = {best_val_f1:.4f}")
    print(f"  Démo Gradio    : python demo.py")
    print(f"{'═'*60}\n")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
