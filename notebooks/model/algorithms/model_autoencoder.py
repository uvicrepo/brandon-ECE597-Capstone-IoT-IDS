#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)


# In[15]:


class PacketAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), latent_dim=8):
        super(PacketAutoencoder, self).__init__()

        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# In[16]:


def build_autoencoder(
    input_dim,
    hidden_dims=(64, 32, 16),   # encoder hidden layer sizes
    latent_dim=8,                # bottleneck / latent feature size
    random_state=42
):
    if random_state is not None:
        torch.manual_seed(random_state)
    return PacketAutoencoder(input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)


# In[17]:


def train_autoencoder(
    model,
    X_train_tensor,
    num_epochs=15,
    batch_size=512,
    lr=0.003,
    verbose=True
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(
        X_train_tensor, batch_size=batch_size, shuffle=True
    )

    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(X_train_tensor)
        loss_history.append(epoch_loss)

        if verbose and ((epoch + 1) % 3 == 0 or epoch == 0):
            print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {epoch_loss:.6f}")

    return model, loss_history


# In[18]:


def compute_reconstruction_errors(model, X_tensor):
    model.eval()
    with torch.no_grad():
        reconstructed, latent = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    return errors, latent.numpy()


# In[19]:


# Find the reconstruction-error threshold that maximizes F1 on a validation set.
def _tune_threshold(errors_val, y_val_binary, n_candidates=300):
    # 300 evenly spaced percentiles across the validation error range
    candidates = np.percentile(errors_val, np.linspace(1, 99, n_candidates))

    best_f1, best_thresh = 0.0, 0.0
    for t in candidates:
        y_hat = (errors_val > t).astype(int)  # above threshold = Attack
        f = f1_score(y_val_binary, y_hat, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, best_f1


# In[20]:


def evaluate_metrics(val, pred, threshold, scores_val):
    acc = accuracy_score(val, pred)
    f1 = f1_score(val, pred, average="binary", zero_division=0)

    report = classification_report(
        val, pred,
        target_names=["Benign", "Attack"],
        zero_division=0
    )
    try:
        # Higher reconstruction error = more anomalous, so no sign-flip needed here
        # (unlike Isolation Forest's decision_function, where higher = more normal)
        roc_auc  = roc_auc_score(val, scores_val)
        avg_prec = average_precision_score(val, scores_val)
    except Exception:
        roc_auc = avg_prec = None

    metrics = {
        "accuracy": acc,
        "f1_binary": f1,
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "threshold": threshold,
        "report": report
    }

    """ print(
        f"\nAccuracy: {acc:.4f} | F1 (Attack): {f1:.4f} | "
        f"ROC-AUC: {roc_auc:.4f}"
    )
    print(report) """
    return metrics


# In[21]:


def plot(val, pred, scores_val, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Confusion matrix
    cm = confusion_matrix(val, pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ROC curve
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(val, scores_val)
        axes[1].plot(fpr, tpr, color="crimson", lw=2,
                     label=f"ROC AUC = {metrics['roc_auc']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

    # Reconstruction error distribution by class
    val = np.array(val)
    scores_val = np.array(scores_val)
    axes[2].hist(scores_val[val == 0], bins=50, alpha=0.6, label="Benign", color="steelblue")
    axes[2].hist(scores_val[val == 1], bins=50, alpha=0.6, label="Attack", color="crimson")
    axes[2].axvline(metrics["threshold"], color="black", linestyle="--", label="Threshold")
    axes[2].set_xlabel("Reconstruction Error")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Reconstruction Error Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# In[22]:


def train(
    X_train, y_train,
    X_val, y_val,
    normal_label="benign",
    model_kwargs: dict = None,
    train_kwargs: dict = None,
    tune_threshold=True,
    save_path: str = None
):
    print("===========")
    print("AUTOENCODER")
    print("===========")
    model_kwargs = model_kwargs or {}
    train_kwargs = train_kwargs or {}

    # Filter to benign-only training, same pattern as the Isolation Forest module
    mask_benign = np.array(y_train) == normal_label
    X_fit = np.array(X_train)[mask_benign]
    n_total  = len(y_train)
    n_benign = mask_benign.sum()
    n_attack = n_total - n_benign
    """
    print(
        f"Training on {n_benign:,} benign samples "
        f"(excluded {n_attack:,} attack samples)."
    )
    """

    X_fit_tensor = torch.tensor(np.array(X_fit), dtype=torch.float32)
    X_val_tensor = torch.tensor(np.array(X_val), dtype=torch.float32)

    input_dim = X_fit_tensor.shape[1]
    model = build_autoencoder(input_dim, **model_kwargs)

    #print("Training autoencoder on benign data...")
    model, loss_history = train_autoencoder(model, X_fit_tensor, **train_kwargs)

    # Binary labels for evaluation: 0 = Benign, 1 = Attack
    y_val_binary = (np.array(y_val) != normal_label).astype(int)

    # Reconstruction error on validation set
    errors_val, _ = compute_reconstruction_errors(model, X_val_tensor)

    # Threshold tuning
    if tune_threshold:
        threshold, best_f1 = _tune_threshold(errors_val, y_val_binary, n_candidates=300)
        print(f"Threshold tuned: {threshold:.6f}  (val F1={best_f1:.4f})")
    else:
        threshold = float(np.percentile(errors_val, 95))
        print(f"Using default threshold (95th percentile of val errors): {threshold:.6f}")

    # Evaluate
    y_pred_binary = (errors_val > threshold).astype(int)
    metrics = evaluate_metrics(y_val_binary, y_pred_binary, threshold, errors_val)
    #plot(y_val_binary, y_pred_binary, errors_val, metrics)

    print("Autoencoder done training...")
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": int(input_dim),
            "model_kwargs": dict(model_kwargs),
            "threshold": float(threshold),
            "normal_label": str(normal_label),
        }, save_path)
        print(f"Saved > {save_path}")

    return model, threshold, metrics


# In[23]:


def predict(
    model,          # trained PacketAutoencoder
    X,              # feature array
    threshold: float,  # reconstruction-error cutoff from train()
    return_scores=False
):
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    errors, _ = compute_reconstruction_errors(model, X_tensor)
    attack_mask = errors > threshold
    labels = np.where(attack_mask, "Attack", "Benign")
    if return_scores:
        return labels, attack_mask, errors
    return labels, attack_mask


# In[24]:


# How to call

# model, threshold, metrics, loss_history = train(
#     X_train, y_train,
#     X_val, y_val,
#     normal_label="benign",
#     model_kwargs={"hidden_dims": (64, 32, 16), "latent_dim": 8, "random_state": 42},
#     train_kwargs={"num_epochs": 15, "batch_size": 512, "lr": 0.003},
#     tune_threshold=True,
#     save_path="models/packet_autoencoder.pt"
# )

# labels, attack_mask, scores = predict(model, X_test, threshold, return_scores=True)


# In[ ]:





# In[ ]:




