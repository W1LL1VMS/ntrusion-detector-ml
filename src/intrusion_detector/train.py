"""Entraînement du modèle final et sauvegarde sur disque.

Usage : python -m intrusion_detector.train
"""

import time
import warnings

import joblib
import sklearn

from intrusion_detector.data import PROJECT_ROOT, load_train, split_features_target
from intrusion_detector.models import BALANCED_SUFFIX, candidate_models

# Modèle retenu à l'étape 6, par validation croisée sur l'entraînement uniquement
FINAL_MODEL_NAME = "Gradient boosting" + BALANCED_SUFFIX
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"


def train_final_model(train_df=None):
    """Entraîne le modèle retenu sur toutes les données d'entraînement, pour les cinq familles."""
    if train_df is None:
        train_df = load_train()
    X, y = split_features_target(train_df, target="category")
    model = candidate_models(balanced=True)[FINAL_MODEL_NAME]
    return model.fit(X, y.astype(str))


def save_model(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "name": FINAL_MODEL_NAME, "sklearn_version": sklearn.__version__},
        path,
    )


def load_model(path=MODEL_PATH):
    """Charge le modèle sauvegardé.

    Attention : un fichier joblib peut exécuter du code à l'ouverture. On ne charge
    donc que des fichiers produits soi-même, jamais un fichier reçu d'une source inconnue.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {path}\nEntraîne-le avec : python -m intrusion_detector.train"
        )
    payload = joblib.load(path)
    if payload["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"Modèle entraîné avec scikit-learn {payload['sklearn_version']}, "
            f"version installée : {sklearn.__version__}. Réentraîne-le si les résultats semblent incohérents.",
            UserWarning,
        )
    return payload["model"]


def main():
    started = time.perf_counter()
    train_df = load_train()
    model = train_final_model(train_df)
    save_model(model)
    print(f"{FINAL_MODEL_NAME} entraîné sur {len(train_df):,} connexions".replace(",", " "))
    print(f"Modèle enregistré dans {MODEL_PATH} ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
