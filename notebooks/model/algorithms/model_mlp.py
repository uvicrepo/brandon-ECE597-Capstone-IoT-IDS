#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# In[ ]:


def build_mlp(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
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


def plot(y_true, y_pred, label_encoder, loss_curve=None):
    nplots = 2 if loss_curve is not None else 1
    fig, axes = plt.subplots(1, nplots, figsize=(7 * nplots, 5))
    if nplots == 1:
        axes = [axes]

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

    if loss_curve is not None:
        axes[1].plot(loss_curve, color="mediumpurple", lw=2)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss Curve")

    plt.tight_layout()
    plt.show()


# In[ ]:


def train(
    X_train, y_train,
    X_val, y_val,
    normal_label="Benign",
    model_kwargs=None,
    scale=True,
    save_path=None,
):
    print("\nMLP\n" + "=" * 11)
    kwargs = model_kwargs or {}

    X_train_arr = np.asarray(X_train)
    X_val_arr = np.asarray(X_val)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(np.asarray(y_train))
    y_val_enc = le.transform(np.asarray(y_val))

    print(f"Training on {len(y_train_enc):,} samples across {len(le.classes_)} classes.")
    print("Classes:", list(le.classes_))

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_arr = scaler.fit_transform(X_train_arr)
        X_val_arr = scaler.transform(X_val_arr)
        print("Features scaled with StandardScaler.")

    model = build_mlp(**kwargs)
    model.fit(X_train_arr, y_train_enc)
    print(f"MLP fitted ({len(model.loss_curve_)} iterations).")

    y_pred_enc = model.predict(X_val_arr)
    metrics = evaluate_metrics(y_val_enc, y_pred_enc, le)
    #plot(y_val_enc, y_pred_enc, le, loss_curve=model.loss_curve_)

    print("MLP done training...")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "label_encoder": le,
                "normal_label": normal_label,
            },
            save_path,
        )
        print(f"Saved > {save_path}")

    return model, scaler, metrics


# In[ ]:


def predict(model, X, scaler=None, label_encoder=None, return_scores=False):
    X_arr = np.asarray(X)
    if scaler is not None:
        X_arr = scaler.transform(X_arr)

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

