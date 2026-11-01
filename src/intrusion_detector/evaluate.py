"""Évaluation des modèles : validation croisée, rapports par famille, matrice de confusion."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from intrusion_detector.data import CATEGORIES

RANDOM_STATE = 42

# zero_division=0 : un modèle qui ne prédit jamais « attaque » a une précision de 0, pas une erreur
BINARY_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def make_cv(n_splits=5, random_state=RANDOM_STATE):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_binary(model, X, y, n_splits=5, n_jobs=-1):
    """Validation croisée stratifiée : moyenne et écart-type de chaque métrique."""
    results = cross_validate(
        model, X, y, cv=make_cv(n_splits), scoring=BINARY_SCORING, n_jobs=n_jobs
    )
    scores = {name: results[f"test_{name}"] for name in BINARY_SCORING}
    return pd.DataFrame(
        {
            "moyenne": {name: values.mean() for name, values in scores.items()},
            "écart-type": {name: values.std() for name, values in scores.items()},
        }
    )


def cross_val_predictions(model, X, y, n_splits=5, n_jobs=-1):
    """Prédictions « hors pli » : chaque connexion est prédite par un modèle qui ne l'a jamais vue."""
    y = y.astype(str)
    predictions = cross_val_predict(model, X, y, cv=make_cv(n_splits), n_jobs=n_jobs)
    return y, predictions


def per_class_report(y_true, y_pred, labels=CATEGORIES):
    """Précision, rappel et F1 pour chaque famille, plus la moyenne non pondérée (macro)."""
    labels = list(labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    table = pd.DataFrame(report).T.loc[labels + ["macro avg"], ["precision", "recall", "f1-score", "support"]]
    table["support"] = table["support"].astype(int)
    return table.rename(columns={"f1-score": "f1"})


def family_summary(predictions):
    """Rappel par famille et F1 macro pour plusieurs modèles : une ligne par modèle.

    `predictions` associe le nom d'un modèle à son couple (vraies familles, familles prédites).
    """
    rows = {}
    for name, (y_true, y_pred) in predictions.items():
        report = per_class_report(y_true, y_pred)
        row = report.loc[CATEGORIES, "recall"].to_dict()
        row["F1 macro"] = report.loc["macro avg", "f1"]
        rows[name] = row
    return pd.DataFrame(rows).T


def plot_confusion_matrix(y_true, y_pred, labels=CATEGORIES, title="", ax=None):
    """Matrice de confusion : chaque ligne montre où vont les connexions d'une vraie famille."""
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_totals, out=np.zeros(matrix.shape, dtype=float), where=row_totals > 0)
    annotations = np.array(
        [[f"{share:.1%}\n({count:,})".replace(",", " ") for share, count in zip(share_row, count_row)]
         for share_row, count_row in zip(shares, matrix)]
    )
    ax = sns.heatmap(
        shares,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Famille prédite")
    ax.set_ylabel("Vraie famille")
    ax.set_title(title)
    return ax
