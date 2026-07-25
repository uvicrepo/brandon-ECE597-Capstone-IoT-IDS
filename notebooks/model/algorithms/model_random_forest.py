#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# In[ ]:


def build_random_forest(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )


# In[ ]:


def evaluate_metrics(y_true, y_pred, label_encoder):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "report": report,
    }

    print(f"\nAccuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    print(report)
    return metrics


# In[ ]:


def plot(y_true, y_pred, label_encoder, model=None, X_train=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    if model is not None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_k = min(20, len(importances))
        top_idx = np.argsort(importances)[-top_k:]

        if X_train is not None and hasattr(X_train, "columns"):
            feature_names = np.array(X_train.columns)[top_idx]
        else:
            feature_names = np.array([f"feature_{i}" for i in top_idx])

        axes[1].barh(range(top_k), importances[top_idx], color="steelblue")
        axes[1].set_yticks(range(top_k))
        axes[1].set_yticklabels(feature_names)
        axes[1].set_title(f"Top {top_k} Feature Importances")
        axes[1].set_xlabel("Importance")
    else:
        axes[1].set_title("Feature Importances")
        axes[1].text(0.5, 0.5, "Not available", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.show()


# In[ ]:


def train(
    X_train, y_train,
    X_val, y_val,
    normal_label="Benign",
    model_kwargs=None,
    save_path=None,
):
    print("\nRANDOM FOREST\n" + "=" * 13)
    kwargs = model_kwargs or {}

    X_train_arr = np.asarray(X_train)
    X_val_arr = np.asarray(X_val)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(np.asarray(y_train))
    y_val_enc = le.transform(np.asarray(y_val))

    print(f"Training on {len(y_train_enc):,} samples across {len(le.classes_)} classes.")
    print("Classes:", list(le.classes_))

    model = build_random_forest(**kwargs)
    model.fit(X_train_arr, y_train_enc)

    y_pred_enc = model.predict(X_val_arr)
    metrics = evaluate_metrics(y_val_enc, y_pred_enc, le)
    #plot(y_val_enc, y_pred_enc, le, model=model, X_train=X_train)

    print("Random forest done training...")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "label_encoder": le,
                "normal_label": normal_label,
            },
            save_path,
        )
        print(f"Saved > {save_path}")

    return model, metrics, le


# In[ ]:


def predict(model, X, label_encoder=None, return_scores=False):
    X_arr = np.asarray(X)
    proba = model.predict_proba(X_arr)
    pred_enc = model.predict(X_arr)

    if label_encoder is not None:
        labels = label_encoder.inverse_transform(pred_enc)
    else:
        labels = pred_enc

    pred_score = np.max(proba, axis=1)

    if return_scores:
        return labels, pred_enc, pred_score, proba
    return labels, pred_enc

