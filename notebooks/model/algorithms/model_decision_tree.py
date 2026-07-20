#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve
)


# In[ ]:


def build_decision_tree(
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features=None,
    criterion: str = "gini",
    class_weight: str = "balanced",
    random_state: int = 42
):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
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


def plot(val, pred, proba_attack, metrics, model=None, feature_names=None):
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
        fpr, tpr, _ = roc_curve(val, proba_attack)
        axes[1].plot(fpr, tpr, color="darkorange", lw=2,
                     label=f"ROC AUC = {metrics['roc_auc']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

    # Feature importances (top 20)
    importances = metrics.get("feature_importances")
    if importances is not None:
        top_idx = np.argsort(importances)[-20:]
        axes[2].barh(range(20), importances[top_idx], color="darkorange")
        labels = (
            [feature_names[i] for i in top_idx]
            if feature_names is not None
            else [str(i) for i in top_idx]
        )
        axes[2].set_yticks(range(20))
        axes[2].set_yticklabels(labels)
        axes[2].set_title("Top 20 Feature Importances")
        axes[2].set_xlabel("Importance")

    plt.tight_layout()
    plt.show()


# In[ ]:


def train(
    X_train, y_train,
    X_val,   y_val,
    normal_label: str = "Benign",
    model_kwargs: dict = None,
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

    feature_names = (
        list(X_train.columns) if hasattr(X_train, "columns") else None
    )

    model = build_decision_tree(**kwargs)
    model.fit(np.array(X_train), y_train_binary)
    print("Decision Tree fitted.")

    proba_attack  = model.predict_proba(np.array(X_val))[:, 1]
    y_pred_binary = model.predict(np.array(X_val))

    metrics = evaluate_metrics(y_val_binary, y_pred_binary, proba_attack)
    metrics["feature_importances"] = model.feature_importances_

    plot(y_val_binary, y_pred_binary, proba_attack, metrics,
         model=model, feature_names=feature_names)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(
            {"model": model, "normal_label": normal_label},
            save_path
        )
        print(f"Saved > {save_path}")

    return model, metrics


# In[ ]:


def predict(model, X, return_scores: bool = False):
    X_arr         = np.array(X)
    proba_attack  = model.predict_proba(X_arr)[:, 1]
    attack_mask   = model.predict(X_arr).astype(bool)
    labels        = np.where(attack_mask, "Attack", "Benign")

    if return_scores:
        return labels, attack_mask, proba_attack
    return labels, attack_mask

