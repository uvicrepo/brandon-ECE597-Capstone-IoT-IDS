#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# In[ ]:


def build_kmeans(
    n_clusters=2,
    random_state=42,
    n_init="auto",
):
    return KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )


# In[ ]:


def evaluate_metrics(model, X):
    labels = model.predict(X)
    inertia = model.inertia_
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    ch = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0.0

    metrics = {
        "inertia": inertia,
        "silhouette": sil,
        "calinski_harabasz": ch,
    }

    print(f"Inertia: {inertia:.4f} | Silhouette: {sil:.4f} | CH: {ch:.4f}")
    return metrics


# In[ ]:


def plot(model, X, X_train=None):
    labels = model.predict(X)
    centers = model.cluster_centers_

    if X.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="tab10", s=25, ax=ax)
        ax.scatter(centers[:, 0], centers[:, 1], c="black", s=200, marker="X")
        ax.set_title("KMeans Clusters")
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        plt.tight_layout()
        plt.show()

    if X_train is not None and hasattr(X_train, "columns"):
        dist = model.transform(np.asarray(X_train))
        min_dist = dist.min(axis=1)
        fi = pd.DataFrame({"feature": X_train.columns, "distance": model.inertia_ / len(X_train)})
        print(fi.head())


# In[ ]:


def train(
    X_train, y_train=None,
    X_val=None, y_val=None,
    normal_label="Benign",
    model_kwargs=None,
    save_path=None,
):
    print("===========")
    print("KMEANS")
    print("===========")

    model_kwargs = model_kwargs or {}

    X_tr = np.asarray(X_train)
    X_v = np.asarray(X_val) if X_val is not None else X_tr

    model = build_kmeans(**model_kwargs)
    model.fit(X_tr)

    metrics = evaluate_metrics(model, X_v)
    # plot(model, X_v, X_train)

    print("KMeans done training")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "normal_label": normal_label,
            },
            save_path,
        )
        print(f"Saved > {save_path}")

    return model, metrics, None


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




