#!/usr/bin/env python
# coding: utf-8

# ### XGBoost

# In[9]:


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder


# In[10]:


def build_xgboost(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_gpu=False,
    num_classes=None,
    random_state=42,
):
    device = "cuda" if use_gpu else "cpu"
    objective = "binary:logistic" if (num_classes is None or num_classes == 2) else "multi:softprob"
    eval_metric = "logloss" if objective == "binary:logistic" else "mlogloss"

    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
        eval_metric=eval_metric,
        device=device,
        random_state=random_state,
        n_jobs=-1,
        verbosity=1,
    )

    if objective == "multi:softprob":
        kwargs["num_class"] = num_classes

    return XGBClassifier(**kwargs)


# In[27]:


def evaluate_metrics(model, eval_X, eval_y, label_encoder):
    y_pred = model.predict(eval_X)
    acc = accuracy_score(eval_y, y_pred)
    f1 = f1_score(eval_y, y_pred, average="macro", zero_division=0)
    report = classification_report(
        eval_y,
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


# In[39]:


def plot(model, eval_X, eval_y, X_train, label_encoder):
    y_pred = model.predict(eval_X)
    cm = confusion_matrix(eval_y, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax,
    )
    ax.set_title("XGBoost — Attack Classification Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if hasattr(X_train, "columns"):
        fi_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=fi_df, x="importance", y="feature", ax=ax2)
        ax2.set_title("XGBoost — Top 20 Important Features")
        plt.tight_layout()
        plt.show()


# In[40]:


def train(
    X_train, y_train,
    X_val=None, y_val=None,
    normal_label="Benign",
    model_kwargs=None,
    label_encoder=None,
    save_path=None,
):
    print("===========")
    print("XGBoost")
    print("===========")

    kwargs = model_kwargs or {}

    y_tr = np.asarray(y_train)
    X_tr = np.asarray(X_train)

    if X_val is not None and y_val is not None:
        y_v = np.asarray(y_val)
        X_v = np.asarray(X_val)
    else:
        X_v = y_v = None

    if label_encoder is None:
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
    else:
        le = label_encoder
        y_tr_enc = le.transform(y_tr)

    y_v_enc = le.transform(y_v) if y_v is not None else None

    num_classes = len(le.classes_)
    kwargs.setdefault("num_classes", num_classes)

    model = build_xgboost(**kwargs)

    if X_v is not None:
        model.fit(X_tr, y_tr_enc, eval_set=[(X_v, y_v_enc)], verbose=50)
    else:
        model.fit(X_tr, y_tr_enc)

    eval_X = X_v if X_v is not None else X_tr
    eval_y = y_v_enc if y_v_enc is not None else y_tr_enc

    metrics = evaluate_metrics(model, eval_X, eval_y, le)
    # plot(model, eval_X, eval_y, X_train, le)

    print("XGBoost done training...")

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


# In[ ]:




