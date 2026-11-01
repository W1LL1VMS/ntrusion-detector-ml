import pytest

from intrusion_detector.data import (
    ATTACK_CATEGORIES,
    CATEGORICAL_FEATURES,
    CATEGORIES,
    COLUMNS,
    DEFAULT_DATA_DIR,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    TEST_FILE,
    TRAIN_FILE,
    load_nslkdd,
    load_test,
    load_train,
    split_features_target,
)

DATASET_PRESENT = (DEFAULT_DATA_DIR / TRAIN_FILE).exists() and (DEFAULT_DATA_DIR / TEST_FILE).exists()


def write_csv(path, labels):
    """Écrit un petit fichier au format NSL-KDD : une connexion valide par étiquette."""
    categorical_values = {"protocol_type": "tcp", "service": "http", "flag": "SF"}
    rows = []
    for label in labels:
        values = [categorical_values.get(name, "0") for name in FEATURE_NAMES]
        rows.append(",".join(values + [label, "20"]))
    path.write_text("\n".join(rows) + "\n")


def test_column_layout():
    assert len(FEATURE_NAMES) == 41
    assert len(COLUMNS) == 43
    assert len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) == 41


def test_every_attack_belongs_to_a_known_category():
    assert set(ATTACK_CATEGORIES.values()) == set(CATEGORIES)


def test_labels_are_mapped_to_categories(tmp_path):
    path = tmp_path / "sample.txt"
    write_csv(path, ["normal", "neptune", "satan", "guess_passwd", "rootkit"])

    df = load_nslkdd(path)

    assert list(df["category"]) == ["normal", "dos", "probe", "r2l", "u2r"]
    assert list(df["is_attack"]) == [0, 1, 1, 1, 1]


def test_unknown_label_is_rejected(tmp_path):
    path = tmp_path / "sample.txt"
    write_csv(path, ["normal", "attaque_inconnue"])

    with pytest.raises(ValueError, match="attaque_inconnue"):
        load_nslkdd(path)


def test_missing_file_gives_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="intrusion_detector.download"):
        load_nslkdd(tmp_path / "absent.txt")


def test_split_never_uses_label_or_difficulty(tmp_path):
    path = tmp_path / "sample.txt"
    write_csv(path, ["normal", "neptune"])
    df = load_nslkdd(path)

    X, y = split_features_target(df)

    assert list(X.columns) == FEATURE_NAMES
    assert "difficulty" not in X.columns
    assert "label" not in X.columns
    assert list(y) == [0, 1]


def test_split_rejects_an_unknown_target(tmp_path):
    path = tmp_path / "sample.txt"
    write_csv(path, ["normal"])

    with pytest.raises(ValueError):
        split_features_target(load_nslkdd(path), target="label")


@pytest.mark.skipif(not DATASET_PRESENT, reason="jeu de données non téléchargé")
def test_real_dataset_matches_published_figures():
    train, test = load_train(), load_test()

    assert len(train) == 125973
    assert len(test) == 22544
    assert train["category"].value_counts().to_dict() == {
        "normal": 67343, "dos": 45927, "probe": 11656, "r2l": 995, "u2r": 52,
    }
    assert test["category"].value_counts().to_dict() == {
        "normal": 9711, "dos": 7458, "probe": 2421, "r2l": 2754, "u2r": 200,
    }
    assert len(set(test["label"]) - set(train["label"])) == 17
