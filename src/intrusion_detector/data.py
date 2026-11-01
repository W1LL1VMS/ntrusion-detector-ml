"""Chargement et description du jeu de données NSL-KDD.

Les fichiers KDDTrain+.txt et KDDTest+.txt sont des CSV sans en-tête : 41
caractéristiques décrivant une connexion réseau, puis l'étiquette (« normal » ou
le nom de l'attaque) et un niveau de difficulté.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"

FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURE_NAMES + ["label", "difficulty"]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [name for name in FEATURE_NAMES if name not in CATEGORICAL_FEATURES]

CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]

_ATTACKS_BY_CATEGORY = {
    "dos": [
        "back", "land", "neptune", "pod", "smurf", "teardrop",
        "apache2", "udpstorm", "processtable", "mailbomb",
    ],
    "probe": ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "r2l": [
        "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
        "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
        "worm", "sendmail", "named",
    ],
    "u2r": [
        "buffer_overflow", "loadmodule", "rootkit", "perl",
        "sqlattack", "xterm", "ps", "httptunnel",
    ],
}

# Nom de l'attaque -> famille d'attaque
ATTACK_CATEGORIES = {"normal": "normal"} | {
    attack: category
    for category, attacks in _ATTACKS_BY_CATEGORY.items()
    for attack in attacks
}


def load_nslkdd(path):
    """Charge un fichier NSL-KDD et ajoute les colonnes `category` et `is_attack`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Télécharge le jeu de données avec : python -m intrusion_detector.download"
        )

    df = pd.read_csv(path, header=None, names=COLUMNS)

    unknown = sorted(set(df["label"]) - set(ATTACK_CATEGORIES))
    if unknown:
        raise ValueError(f"Étiquettes inconnues dans {path.name} : {unknown}")

    df["category"] = pd.Categorical(df["label"].map(ATTACK_CATEGORIES), categories=CATEGORIES)
    df["is_attack"] = (df["category"] != "normal").astype(int)
    return df


def load_train(data_dir=DEFAULT_DATA_DIR):
    return load_nslkdd(Path(data_dir) / TRAIN_FILE)


def load_test(data_dir=DEFAULT_DATA_DIR):
    return load_nslkdd(Path(data_dir) / TEST_FILE)


def split_features_target(df, target="is_attack"):
    """Sépare les caractéristiques de la cible.

    `target` vaut « is_attack » (normal contre attaque) ou « category » (5 classes).
    La colonne `difficulty` n'est jamais une caractéristique : elle est calculée à
    partir des résultats de classifieurs, l'utiliser reviendrait à tricher.
    """
    if target not in ("is_attack", "category"):
        raise ValueError("target doit valoir 'is_attack' ou 'category'")
    return df[FEATURE_NAMES], df[target]
