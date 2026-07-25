#!/usr/bin/env python
# coding: utf-8

# ### ISOLATION FOREST

# In[81]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)


# In[82]:


# Build and return a configured IsolationForest.

def build_isolation_forest(
    n_estimators = 200, #Number of isolation trees.
    max_samples = 256, # Subsampling size per tree. 256 is the original paper default;
    contamination = "auto", 
    max_features: float = 1.0,
    random_state: int = 42
):
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state
    )


# In[83]:


# Find best threshold

def _tune_threshold(model, X_val, y_val_binary, n_candidates = 300):
    scores = model.decision_function(X_val)   # higher = more normal

    # Generate 300 score values to test potential cut-off points
    # i.e 300 evenly spaced percentages from 1% to 99%
    candidates = np.percentile(scores, np.linspace(1, 99, n_candidates))

    # initialize highest accuracy score & top threshold
    best_f1, best_thresh = 0.0, 0.0
    for t in candidates:
        y_hat = (scores < t).astype(int) # below threshold = Attack
        f = f1_score(y_val_binary, y_hat, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, best_f1


# In[84]:


def evaluate_metrics(val, pred, threshold, scores_val):

    acc = accuracy_score(val, pred)
    f1 = f1_score(val, pred, average="binary", zero_division=0)

    report = classification_report(
        val, pred,
        target_names=["Benign", "Attack"],
        zero_division=0
    )

    try:
        roc_auc  = roc_auc_score(val, -scores_val)
        avg_prec = average_precision_score(val, -scores_val)
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

    """ print (
        f"\nAccuracy: {acc:.4f} | F1 (Attack): {f1:.4f} | "
          f"ROC-AUC: {roc_auc:.4f}"
    )

    print(report) """

    return metrics


# In[ ]:


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
        fpr, tpr, _ = roc_curve(val, -scores_val)
        axes[1].plot(fpr, tpr, color="crimson", lw=2,
                     label=f"ROC AUC = {metrics['roc_auc']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()


# In[86]:


def train(
    X_train, y_train,
    X_val, y_val,
    normal_label = "Benign",
    model_kwargs: dict = None,
    tune_threshold = True,
    save_path: str = None
):
    print("================")
    print("ISOLATION FOREST")
    print("================")
    kwargs = model_kwargs or {}

    # Make binary for normal_label = True, others = False
    mask_benign = np.array(y_train) == normal_label

    # Filter for only benign
    X_fit = np.array(X_train)[mask_benign]
    n_total  = len(y_train)
    n_benign = mask_benign.sum() #only counts the True
    n_attack = n_total - n_benign
    print(
        f"Training on {n_benign:,} benign samples "
        f"(excluded {n_attack:,} attack samples)."
    )

    model = build_isolation_forest(**kwargs)
    model.fit(X_fit)

    print("Isolation Forest fitted.")

    # Binary labels for evaluation: 0=Benign, 1=Attack
    y_val_binary = (np.array(y_val) != normal_label).astype(int)

    # Threshold tuning
    if tune_threshold:
        threshold, best_f1 = _tune_threshold(
            model, np.array(X_val), y_val_binary, 300
        )
        print(f"Threshold tuned: {threshold:.5f}  (val F1={best_f1:.4f})")
    else:
        threshold = 0.0
        print(f"Using default threshold: {threshold}")

    # Evaluate
    scores_val = model.decision_function(np.array(X_val))
    y_pred_binary = (scores_val < threshold).astype(int)

    metrics = evaluate_metrics(y_val_binary, y_pred_binary, threshold, scores_val)

    # plot(y_val_binary, y_pred_binary, scores_val, metrics)

    print("Isolation forest done training...")

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(
            {"model": model, "threshold": threshold, "normal_label": normal_label},
            save_path
        )
        print(f"Saved > {save_path}")

    return model, threshold, metrics



# In[87]:


def predict(
    model, #Fitted IsolationForest
    X, #Feature array
    threshold: float, # Decision score cutoff from train()
    return_scores = False #Return raw anomaly scores if True
):
    X_arr  = np.array(X)
    scores = model.decision_function(X_arr)
    attack_mask = scores < threshold #Boolean mask — True where Attack was predicted
    labels = np.where(attack_mask, "Attack", "Benign") #raw decision function scores

    if return_scores:
        return labels, attack_mask, scores
    return labels, attack_mask


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




