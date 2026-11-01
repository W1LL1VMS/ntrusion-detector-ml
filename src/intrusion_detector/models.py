"""Définition des modèles comparés."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intrusion_detector.preprocessing import build_preprocessor


def make_pipeline(estimator):
    """Associe le prétraitement et un modèle dans un seul objet."""
    return Pipeline([("preprocessing", build_preprocessor()), ("model", estimator)])


def baseline_models():
    """Une référence naïve et un premier vrai modèle."""
    return {
        "Référence (classe majoritaire)": make_pipeline(DummyClassifier(strategy="most_frequent")),
        "Régression logistique": make_pipeline(LogisticRegression(max_iter=1000)),
    }
