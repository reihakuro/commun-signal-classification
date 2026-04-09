import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import (
    CUTMIX_ALPHA,
    DEVICE,
    EPOCHS,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    GROUP_ID,
    IMAGE_SIZE,
    MAX_LR,
    MIXUP_ALPHA,
)
from dataset import get_dataloaders
from model import BasicCNN
from utils import cutmix_data, mixed_criterion, mixup_data, set_seed


def main():

    set_seed(42)
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    loaders, class_names, num_classes = get_dataloaders()
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    train_eval_loader = loaders["train_eval"]
    full_loader = loaders["full"]

    model = BasicCNN(num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.05)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=1000,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    def train_model_phase1():
        best_val_acc = 0.0
        final_val_preds = []
        final_val_labels = []
        best_state_dict = None

        print("\n[INFO] Phase 1: training on 80% data...")
        print("-" * 78)

        for epoch in range(EPOCHS):
            # ── TRAINING PHASE ────────────────────────────────────────────────
            model.train()
            running_loss = 0.0
            all_train_preds = []
            all_train_labels = []

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Randomly apply MixUp or CutMix (50 / 50 per batch)
                if np.random.rand() < 0.5:
                    aug_images, y_a, y_b, lam = mixup_data(images, labels, alpha=MIXUP_ALPHA)
                else:
                    aug_images, y_a, y_b, lam = cutmix_data(images, labels, alpha=CUTMIX_ALPHA)

                optimizer.zero_grad()
                outputs = model(aug_images)
                loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

                with torch.no_grad():
                    orig_outputs = model(images)
                _, predicted = torch.max(orig_outputs, 1)
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader)
            train_acc = accuracy_score(all_train_labels, all_train_preds)

            # ── VALIDATION PHASE ──────────────────────────────────────────────
            model.eval()
            val_running_loss = 0.0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

            val_loss = val_running_loss / len(val_loader)
            val_acc = accuracy_score(all_val_labels, all_val_preds)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Save checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_val_preds = all_val_preds
                final_val_labels = all_val_labels
                best_state_dict = copy.deepcopy(model.state_dict())

            print(
                f"Epoch [{epoch + 1:03d}/{EPOCHS}] | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
                f"Best Val: {best_val_acc:.4f}"
            )

        # Restore weights
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f"\n[INFO] Restored best checkpoint → Val Acc = {best_val_acc:.4f}")

        return final_val_labels, final_val_preds

    final_labels, final_preds = train_model_phase1()

    # ==========================================
    # TRAINING LOOP - PHASE 2 (FINE-TUNE)
    # ==========================================
    print(f"\n{'=' * 78}")
    print(f"[INFO] Phase 2: fine-tuning on 100% data for {FINETUNE_EPOCHS} epochs")
    print(f"[INFO] LR = {FINETUNE_LR}  (CosineAnnealingLR, no warm-up)")
    print(f"{'=' * 78}")

    ft_optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.05)
    ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=FINETUNE_EPOCHS, eta_min=FINETUNE_LR / 100)

    for ft_epoch in range(FINETUNE_EPOCHS):
        model.train()
        ft_loss_sum = 0.0

        for images, labels in full_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if np.random.rand() < 0.5:
                aug_images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
            else:
                aug_images, y_a, y_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)

            ft_optimizer.zero_grad()
            outputs = model(aug_images)
            loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            ft_optimizer.step()
            ft_loss_sum += loss.item()

        ft_scheduler.step()
        avg_loss = ft_loss_sum / len(full_loader)
        cur_lr = ft_scheduler.get_last_lr()[0]
        print(f"  Fine-tune [{ft_epoch + 1:02d}/{FINETUNE_EPOCHS}] | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}")

    print("[INFO] Phase 2 complete.")

    # ==========================================
    # EVALUATION ON TRAINING SET
    # ==========================================
    print("\n[INFO] Evaluating on TRAINING SET (no augmentation)...")
    model.eval()
    train_true_labels = []
    train_pred_labels = []

    with torch.no_grad():
        for images, labels in train_eval_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(predicted.cpu().numpy())

    # ==========================================
    # METRICS REPORT
    # ==========================================
    print("\n" + "=" * 78)
    print("[INFO] DETAILED CLASSIFICATION REPORT (Training Set)")
    print("=" * 78)

    overall_acc = accuracy_score(train_true_labels, train_pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        train_true_labels, train_pred_labels, zero_division=0
    )

    print(f"Overall Accuracy (Training Set): {overall_acc:.4f}\n")
    hdr = f"{'Class':<12}  {'Precision':>10}  {'Recall':>10}  {'F1-score':>10}  {'Support':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i, cls in enumerate(class_names):
        print(f"{cls:<12}  {precision[i]:>10.4f}  {recall[i]:>10.4f}  {f1[i]:>10.4f}  {int(support[i]):>8}")
    print("=" * 78)
    print(f"\n[INFO] Best Phase-1 Validation Accuracy: {accuracy_score(final_labels, final_preds):.4f}")

    # ==========================================
    # PLOTTING
    # ==========================================
    epochs_x = range(1, EPOCHS + 1)

    # Loss plot
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_x, history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(epochs_x, history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.title("Training and Validation Loss (Phase 1)", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300)
    plt.close()
    print("[INFO] Saved: loss_plot.png")

    # Accuracy plot
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_x, history["train_acc"], label="Training Accuracy", linewidth=2)
    plt.plot(epochs_x, history["val_acc"], label="Validation Accuracy", linewidth=2)
    plt.axhline(y=0.90, color="red", linestyle="--", alpha=0.7, label="90% target")
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title("Training and Validation Accuracy (Phase 1)", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300)
    plt.close()
    print("[INFO] Saved: accuracy_plot.png")

    # Confusion Matrix
    cm = confusion_matrix(train_true_labels, train_pred_labels)
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
    )
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.title("Confusion Matrix (Training Set)", fontsize=15)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("[INFO] Saved: confusion_matrix.png")

    # ==========================================
    # MODEL EXPORT
    # ==========================================
    model.eval()
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    traced_model = torch.jit.trace(model, example_input)

    model_name = f"{GROUP_ID}_DeepLearningProject_TrainedModel.pt"
    traced_model.save(model_name)
    print("Model saved:", model_name)


if __name__ == "__main__":
    main()
