"""Définition des modèles comparés."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intrusion_detector.preprocessing import build_preprocessor

RANDOM_STATE = 42
BALANCED_SUFFIX = " (poids équilibrés)"


def make_pipeline(estimator):
    """Associe le prétraitement et un modèle dans un seul objet."""
    return Pipeline([("preprocessing", build_preprocessor()), ("model", estimator)])


def baseline_models():
    """Une référence naïve et un premier vrai modèle."""
    return {
        "Référence (classe majoritaire)": make_pipeline(DummyClassifier(strategy="most_frequent")),
        "Régression logistique": make_pipeline(LogisticRegression(max_iter=1000)),
    }


def candidate_models(balanced=False):
    """Trois algorithmes, avec ou sans pondération des classes.

    `balanced=True` donne plus de poids aux familles rares pendant l'apprentissage
    (chaque erreur sur U2R compte davantage qu'une erreur sur le trafic normal).
    """
    weight = "balanced" if balanced else None
    suffix = BALANCED_SUFFIX if balanced else ""
    return {
        "Régression logistique" + suffix: make_pipeline(
            LogisticRegression(max_iter=1000, class_weight=weight)
        ),
        "Forêt aléatoire" + suffix: make_pipeline(
            RandomForestClassifier(
                n_estimators=200, class_weight=weight, random_state=RANDOM_STATE, n_jobs=1
            )
        ),
        "Gradient boosting" + suffix: make_pipeline(
            HistGradientBoostingClassifier(class_weight=weight, random_state=RANDOM_STATE)
        ),
    }
