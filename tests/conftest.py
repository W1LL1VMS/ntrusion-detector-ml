import numpy as np
import pytest

from intrusion_detector.data import FEATURE_NAMES, load_nslkdd

LABELS = ["normal", "neptune", "satan", "guess_passwd", "rootkit"]
CATEGORICAL_VALUES = {
    "protocol_type": ["tcp", "udp", "icmp"],
    "service": ["http", "ftp", "smtp", "private"],
    "flag": ["SF", "S0", "REJ"],
}


@pytest.fixture
def sample_df(tmp_path):
    """Petit jeu de données synthétique au format NSL-KDD : 50 connexions, 10 par étiquette."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(50):
        values = []
        for name in FEATURE_NAMES:
            if name in CATEGORICAL_VALUES:
                options = CATEGORICAL_VALUES[name]
                values.append(options[int(rng.integers(len(options)))])
            else:
                values.append(f"{rng.random() * 10:.2f}")
        rows.append(",".join(values + [LABELS[i % len(LABELS)], "20"]))

    path = tmp_path / "sample.txt"
    path.write_text("\n".join(rows) + "\n")
    return load_nslkdd(path)
