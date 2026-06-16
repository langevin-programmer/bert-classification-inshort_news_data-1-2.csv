# train.py 

import os
import json 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torch.optim import AdamW 
import wandb
import time
from utils import (CONFIG,
                   compute_metrics,
                   set_seed,plot_class_distribution,
                   init_wandb,
                   log_image_wandb,
                   log_epoch_metrics,
                   plot_confusion_matrix,
                   plot_learning_curves,
                   log_confusion_matrix_wandb,
                   log_final_summary
                   )
import wandb
from model import (load_tokenizer,
                   load_model,
                   save_model
                  )
from dataset import TextClassificationDataset
from transformers import get_linear_schedule_with_warmup
    







# 1 - Chargement et préparation des données 

def load_and_inspect_data(
    data_path: str,
    text_col: str,
    article_col: str,
    label_col: str,
) -> tuple:
    """
    Charge le CSV, concatène titre + article, inspecte la distribution
    des classes et encode les labels.

    La colonne d'entrée BERT = text_col + '. ' + article_col.
    BERT encode les deux segments séparément grâce aux token_type_ids.

    Returns:
        tuple: (texts_combined, labels_int, label_encoder, class_names)
    """
    print(f"\n{'═'*60}")
    print(f'   CHARGEMENT DU DATASET')
    print(f"{'═'*60}")

    df_local = pd.read_csv(data_path)
    print(f'  Fichier    : {data_path}')
    print(f'  Shape brut : {df_local.shape}')

    df_local = df_local[[text_col, article_col, label_col]].dropna()
    df_local[text_col]    = df_local[text_col].astype(str).str.strip()
    df_local[article_col] = df_local[article_col].astype(str).str.strip()
    df_local[label_col]   = df_local[label_col].astype(str).str.strip()
    df_local = df_local[(df_local[text_col] != '') & (df_local[article_col] != '')]
    print(f'  Après nettoyage : {len(df_local):,} exemples')

    #  Concaténation titre + article 
    # Format : "[titre]. [article]" → BERT tokenise avec [CLS] titre [SEP] article [SEP]
    df_local['text_combined'] = df_local[text_col] + '. ' + df_local[article_col]
    print(f'  Colonne combinée créée : text_combined = {text_col} + ". " + {article_col}')

    # Distribution
    print(f'\n   Distribution des classes :')
    dist_local = df_local[label_col].value_counts()
    for cls, cnt in dist_local.items():
        pct = cnt / len(df_local) * 100
        bar = '█' * int(pct / 2)
        print(f'    {cls:<20} {cnt:>5} ({pct:5.1f}%) {bar}')

    ratio_local = dist_local.max() / dist_local.min()
    if ratio_local > 2.0:
        print(f'\n    Déséquilibre détecté (ratio max/min = {ratio_local:.1f}x)')
        print(f'     → Stratégie : split stratifié + F1-macro comme métrique principale')

    # Longueurs approximatives (en mots)
    len_h = df_local[text_col].str.split().str.len()
    len_a = df_local[article_col].str.split().str.len()
    len_c = df_local['text_combined'].str.split().str.len()
    print(f'\n   Longueur approx. (mots) :')
    print(f'    Titre seul   → moy={len_h.mean():.0f}, P95={len_h.quantile(0.95):.0f}')
    print(f'    Article seul → moy={len_a.mean():.0f}, P95={len_a.quantile(0.95):.0f}')
    print(f'    Combiné      → moy={len_c.mean():.0f}, P95={len_c.quantile(0.95):.0f}  ← référence pour max_length')

    # Exemples
    print(f'\n   5 exemples (titre + article) :')
    for i in range(min(5, len(df_local))):
        lbl  = df_local[label_col].iloc[i]
        head = df_local[text_col].iloc[i]
        art  = df_local[article_col].iloc[i][:80]
        print(f'    [{lbl:<12}] Titre: {head[:60]}...')
        print(f'    {"":<14}  Art  : {art}...')

    # Encodage des labels
    le = LabelEncoder()
    labels_int  = le.fit_transform(df_local[label_col].values)
    class_names = list(le.classes_)
    texts_combined = df_local['text_combined'].values.tolist()

    print(f'\n    Mapping label → indice :')
    for idx, name in enumerate(class_names):
        print(f'    {idx} → {name}')

    return texts_combined, labels_int, le, class_names


# print(' load_and_inspect_data défini.')


# 2- Boucle d'entraînement et d'évaluation

def train_epoch(
    model, loader, optimizer, scheduler, criterion, device, epoch_num
) -> tuple:
    """
    Effectue une epoch complète d'entraînement.

    Procédure par batch :
        1. Transfert des tenseurs sur `device`.
        2. Zero_grad pour effacer les gradients accumulés.
        3. Forward pass (logits via BERT).
        4. Calcul de la loss (CrossEntropyLoss).
        5. Backward pass (calcul des gradients).
        6. Gradient clipping (évite l'explosion du gradient).
        7. Mise à jour des paramètres (optimizer.step).
        8. Mise à jour du scheduler.

    Returns:
        tuple: (avg_loss, accuracy, all_preds, all_labels)
    """
    model.train()  # ← Active dropout
    total_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(
        loader, desc=f'   Epoch {epoch_num} [Train]', leave=False, unit='batch'
    )

    for step, batch in enumerate(pbar):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_b       = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits  # [batch_size, num_classes]

        loss = criterion(logits, labels_b)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['clip_grad'])
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels_b.detach().cpu().tolist())

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        wandb.log({'batch/train_loss': loss.item(), 'batch/step': step})

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics['accuracy'], all_preds, all_labels


def eval_epoch(
    model, loader, criterion, device, epoch_num
) -> tuple:
    """
    Évalue le modèle sans mise à jour des gradients.

    Points critiques :
        - model.eval()    : désactive dropout.
        - torch.no_grad() : désactive le graphe de calcul → 3-4× plus rapide.

    Returns:
        tuple: (avg_loss, accuracy, f1_macro, all_preds, all_labels)
    """
    model.eval()  # ← Désactive dropout
    total_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(
        loader, desc=f'   Epoch {epoch_num} [Val  ]', leave=False, unit='batch'
    )

    with torch.no_grad():  # ← Pas de gradient
        for batch in pbar:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_b       = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            loss    = criterion(logits, labels_b)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels_b.detach().cpu().tolist())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics['accuracy'], metrics['f1_macro'], all_preds, all_labels


# print(' Fonctions train_epoch et eval_epoch définies.')

# 3- Lancement de l'entraînement
def main():
    
    # Connexion W&B
    # Décommentez la ligne suivante si vous n'avez pas encore configuré W&B :
    # wandb.login()

    t_start = time.time()
    print(f"\n{'═'*60}")
    print(f'   BERT FINE-TUNING — InShort News Classification')
    print(f'     Dakar Institute of Technology (DIT) — Master 2 IA')
    print(f"{'═'*60}")

    # Seed & device
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n    Device : {device}')

    # W&B
    run = init_wandb(CONFIG['project_wandb'], CONFIG, CONFIG['run_name'])

    # Données
    texts, labels, label_encoder, class_names = load_and_inspect_data(
        CONFIG['dataset_path'],
        CONFIG['text_column'],
        CONFIG['article_column'],
        CONFIG['label_column'],
    )
    CONFIG['num_classes'] = len(class_names)
    CONFIG['class_names'] = class_names
    wandb.config.update({'num_classes': len(class_names), 'class_names': class_names})

    # Graphique distribution
    dist_path = plot_class_distribution(
        list(labels), class_names, save_path='class_distribution.png'
    )
    log_image_wandb('data/class_distribution', dist_path, 'Distribution des classes')

    # Split stratifié 80/20
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=CONFIG['val_split'],
        random_state=CONFIG['seed'],
        stratify=labels,  # Préserve la distribution originale
    )
    print(f'\n   Split : {len(X_train):,} train | {len(X_val):,} val')

    # Tokenizer & Datasets
    print(f'\n   Chargement du tokenizer…')
    tokenizer = load_tokenizer(CONFIG['model_name'])

    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, CONFIG['max_length'])
    val_dataset   = TextClassificationDataset(X_val,   y_val,   tokenizer, CONFIG['max_length'])

    nw = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=True,  num_workers=nw, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=nw, pin_memory=(device.type == 'cuda'))

    # Modèle
    print(f'\n   Chargement du modèle BERT…')
    model = load_model(CONFIG['num_classes'], CONFIG['model_name'])
    model = model.to(device)

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    with open(os.path.join(CONFIG['save_dir'], 'class_names.json'), 'w') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Optimiseur & Scheduler
    # AdamW : Adam avec décroissance de poids découplée (recommandé pour BERT)
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        eps=1e-8,
    )
    total_steps  = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

    # Scheduler linéaire : lr monte pendant warmup_steps, puis décroît linéairement
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f'\n    Optimiseur : AdamW | lr={CONFIG["learning_rate"]} | wd={CONFIG["weight_decay"]}')
    print(f'     Total steps : {total_steps} | Warmup steps : {warmup_steps}')

    # CrossEntropyLoss attend des logits (sans softmax) et des labels entiers
    criterion = nn.CrossEntropyLoss()
    
    # Boucle d'entraînement 
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'val_f1': [],
    }
    best_val_loss    = float('inf')
    best_val_acc     = 0.0
    best_val_f1      = 0.0
    best_val_preds   = []
    best_val_labels  = []

    print(f"\n{'═'*60}")
    print(f"   DÉMARRAGE DE L'ENTRAÎNEMENT ({CONFIG['num_epochs']} epochs)")
    print(f"{'═'*60}\n")

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n  ┌─ Epoch {epoch}/{CONFIG['num_epochs']} {'─'*40}")

        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        val_loss, val_acc, val_f1, val_preds, val_labels_list = eval_epoch(
            model, val_loader, criterion, device, epoch
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1)

        current_lr = scheduler.get_last_lr()[0]
        print(f'  │  Train : loss={train_loss:.4f} | acc={train_acc:.4f}')
        print(f'  │  Val   : loss={val_loss:.4f} | acc={val_acc:.4f} | f1={val_f1:.4f}')
        print(f'  │  LR    : {current_lr:.2e}')

        log_epoch_metrics(epoch, train_loss, val_loss, train_acc, val_acc, val_f1, current_lr)

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_val_acc    = val_acc
            best_val_f1     = val_f1
            best_val_preds  = val_preds
            best_val_labels = val_labels_list
            save_model(model, tokenizer, class_names, CONFIG['save_dir'])
            print(f'  │   Meilleur modèle sauvegardé ! (val_loss={val_loss:.4f})')
        else:
            print(f'  │  (Pas d\'amélioration. Meilleur val_loss: {best_val_loss:.4f})')

        print(f"  └{'─'*50}")
        
        # 4 - Visualisations finales & résumé
        
        print(f"\n{'═'*60}")
        print(f'   GÉNÉRATION DES VISUALISATIONS')
        print(f"{'═'*60}")

        # Matrice de confusion
        cm_path = plot_confusion_matrix(best_val_labels, best_val_preds, class_names)
        log_image_wandb('val/confusion_matrix_img', cm_path, 'Matrice de confusion (meilleur modèle)')
        log_confusion_matrix_wandb(best_val_labels, best_val_preds, class_names)

        # Courbes d'apprentissage
        lc_path = plot_learning_curves(history)
        log_image_wandb('train/learning_curves', lc_path, 'Courbes d\'apprentissage')

        # Résumé W&B
        log_final_summary(best_val_loss, best_val_acc, best_val_f1, class_names)

        # Rapport de classification final
        print(f'\n   Rapport de classification final (meilleur checkpoint) :')
        compute_metrics(best_val_labels, best_val_preds, class_names, verbose=True)

        wandb.finish()

        elapsed = time.time() - t_start
        print(f"\n{'═'*60}")
        print(f'  ENTRAÎNEMENT TERMINÉ en {elapsed/60:.1f} min')
        print(f'  Meilleur checkpoint : \'{CONFIG["save_dir"]}/\'')
        print(f'  best_val_loss  = {best_val_loss:.4f}')
        print(f'  best_val_acc   = {best_val_acc:.4f}')
        print(f'  best_val_f1    = {best_val_f1:.4f}')
        print(f"{'═'*60}")


if __name__== "__main__":
    main()