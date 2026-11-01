"""Prétraitement des caractéristiques, partagé par tous les modèles.

Tout est ajusté uniquement sur les données d'entraînement : c'est le Pipeline
scikit-learn qui garantit qu'aucune information du test ou d'un pli de validation
ne fuit vers l'apprentissage.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from intrusion_detector.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Colonne constante à l'entraînement (voir le notebook d'exploration) : aucune information
DROPPED_FEATURES = ["num_outbound_cmds"]
USED_NUMERIC_FEATURES = [name for name in NUMERIC_FEATURES if name not in DROPPED_FEATURES]


def build_preprocessor():
    """Normalise les variables numériques et encode les variables catégorielles en 0/1.

    Les valeurs catégorielles jamais vues à l'entraînement sont ignorées (toutes les
    colonnes 0/1 de la variable à zéro) au lieu de provoquer une erreur.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), USED_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
