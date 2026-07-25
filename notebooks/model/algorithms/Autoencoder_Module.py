#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)


# In[ ]:


def split_data(X, y=None, train: float = 0.7, val: float = 0.1, test: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    assert abs(train + val + test - 1.0) < 1e-6, "train + val + test must equal 1.0"

    # First split: train vs rest
    rest = val + test
    X_train, X_temp = train_test_split(X, test_size=rest, random_state=random_state)

    # Second split: val vs test
    test_ratio = test / rest
    X_val, X_test = train_test_split(X_temp, test_size=test_ratio, random_state=random_state)

    if y is not None:
        y_train, y_temp = train_test_split(y, test_size=rest, random_state=random_state)
        y_val, y_test = train_test_split(y_temp, test_size=test_ratio, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_val, X_test


# In[2]:


def train_autoencoder(model, X_train, lr: float = 0.003, batch_size: int = 512, num_epochs: int = 15):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

    epoch_losses = []
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
        avg_loss = epoch_loss / len(X_train)
        epoch_losses.append(avg_loss)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.6f}")

    print("Training complete.")
    return epoch_losses


# In[4]:


def extract_latent(model, X):
    model.eval()
    with torch.no_grad():
        reconstructed, latent = model(X)

    reconstruction_errors = torch.mean((X - reconstructed) ** 2, dim=1).numpy()
    latent_features = latent.numpy()

    return latent_features, reconstruction_errors


# In[ ]:


def evaluate_metrics(y_true, y_pred, reconstruction_errors):
    report = classification_report(
        y_true, y_pred,
        target_names=["Benign", "Anomaly"],
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(y_true, reconstruction_errors)
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 20 + " Detection Report " + "=" * 20)
    print(report)
    if roc_auc is not None:
        print(f"Reconstruction-Error AUC-ROC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Benign  (pred Benign): {cm[0][0]}  |  False Positive: {cm[0][1]}")
    print(f"  False Negative:             {cm[1][0]}  |  True Anomaly:   {cm[1][1]}")

    return {"roc_auc": roc_auc, "confusion_matrix": cm, "report": report}


# In[ ]:


def plot(y_true, reconstruction_errors, metrics, epoch_losses=None):
    n_plots = 3 if epoch_losses is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=["Benign", "Anomaly"],
        yticklabels=["Benign", "Anomaly"],
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ROC curve
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_true, reconstruction_errors)
        axes[1].plot(fpr, tpr, color="mediumpurple", lw=2,
                     label=f"ROC AUC = {metrics['roc_auc']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

    # Training loss curve
    if epoch_losses is not None:
        axes[2].plot(epoch_losses, color="mediumpurple", lw=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Reconstruction Loss")
        axes[2].set_title("Training Loss Curve")

    plt.tight_layout()
    plt.show()


# In[ ]:


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims: list = None, latent_dim: int = 8):
        super(Autoencoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        # Encoder
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# In[9]:


# Load your data
X = data/selected_features.csv

# Split it
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, train=0.7, val=0.1, test=0.2)

# Build the autoencoder
model = Autoencoder(input_dim=X_train.shape[1], hidden_dims=[64, 32, 16], latent_dim=8)

# Train it
epoch_losses = train_autoencoder(model, X_train)

# Get latent features and reconstruction errors
latent, errors = extract_latent(model, X_test)

# ... KMeans or thresholding gives you y_pred ...

# Evaluate
metrics = evaluate_metrics(y_test, y_pred, errors)

# Plot
plot(y_test, errors, metrics, epoch_losses)

