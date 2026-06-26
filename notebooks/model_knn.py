#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve
)


# In[ ]:


def build_knn(
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,          # p=2 → Euclidean, p=1 → Manhattan
    algorithm: str = "auto"
):
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p,
        algorithm=algorithm,
        n_jobs=-1
    )


# In[ ]:


def evaluate_metrics(val, pred, proba_attack):
    acc    = accuracy_score(val, pred)
    f1     = f1_score(val, pred, average="binary", zero_division=0)
    report = classification_report(
        val, pred,
        target_names=["Benign", "Attack"],
        zero_division=0
    )

    try:
        roc_auc  = roc_auc_score(val, proba_attack)
        avg_prec = average_precision_score(val, proba_attack)
    except Exception:
        roc_auc = avg_prec = None

    metrics = {
        "accuracy":      acc,
        "f1_binary":     f1,
        "roc_auc":       roc_auc,
        "avg_precision": avg_prec,
        "report":        report
    }

    print(
        f"\nAccuracy: {acc:.4f} | F1 (Attack): {f1:.4f} | "
        f"ROC-AUC: {roc_auc:.4f}"
    )
    print(report)
    return metrics


# In[ ]:


def plot(val, pred, proba_attack, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
        fpr, tpr, _ = roc_curve(val, proba_attack)
        axes[1].plot(fpr, tpr, color="mediumseagreen", lw=2,
                     label=f"ROC AUC = {metrics['roc_auc']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

    plt.tight_layout()
    plt.show()


# In[ ]:


def train(
    X_train, y_train,
    X_val,   y_val,
    normal_label: str = "Benign",
    model_kwargs: dict = None,
    scale: bool = True,
    save_path: str = None
):
    kwargs = model_kwargs or {}

    y_train_binary = (np.array(y_train) != normal_label).astype(int)
    y_val_binary   = (np.array(y_val)   != normal_label).astype(int)

    n_benign = (y_train_binary == 0).sum()
    n_attack = (y_train_binary == 1).sum()
    print(
        f"Training on {len(y_train_binary):,} samples "
        f"({n_benign:,} benign, {n_attack:,} attack)."
    )

    X_train_arr = np.array(X_train)
    X_val_arr   = np.array(X_val)

    # KNN is distance-based so scaling is important
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_arr = scaler.fit_transform(X_train_arr)
        X_val_arr   = scaler.transform(X_val_arr)
        print("Features scaled with StandardScaler.")

    model = build_knn(**kwargs)
    model.fit(X_train_arr, y_train_binary)
    print("KNN fitted.")

    proba_attack  = model.predict_proba(X_val_arr)[:, 1]
    y_pred_binary = model.predict(X_val_arr)

    metrics = evaluate_metrics(y_val_binary, y_pred_binary, proba_attack)

    plot(y_val_binary, y_pred_binary, proba_attack, metrics)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(
            {"model": model, "scaler": scaler, "normal_label": normal_label},
            save_path
        )
        print(f"Saved > {save_path}")

    return model, scaler, metrics


# In[ ]:


def predict(model, X, scaler=None, return_scores: bool = False):
    X_arr = np.array(X)

    if scaler is not None:
        X_arr = scaler.transform(X_arr)

    proba_attack  = model.predict_proba(X_arr)[:, 1]
    attack_mask   = model.predict(X_arr).astype(bool)
    labels        = np.where(attack_mask, "Attack", "Benign")

    if return_scores:
        return labels, attack_mask, proba_attack
    return labels, attack_mask

