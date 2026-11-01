"""Téléchargement du jeu de données NSL-KDD avec vérification d'intégrité (SHA-256).

Usage : python -m intrusion_detector.download
"""

import hashlib
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote

from intrusion_detector.data import DEFAULT_DATA_DIR, TEST_FILE, TRAIN_FILE

# Deux copies publiques indépendantes, essayées dans l'ordre
MIRRORS = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/",
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/",
]

EXPECTED_SHA256 = {
    TRAIN_FILE: "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    TEST_FILE: "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _fetch(url, timeout=60):
    request = urllib.request.Request(url, headers={"User-Agent": "intrusion-detector-ml"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _download_file(filename, expected, target):
    errors = []
    for base in MIRRORS:
        url = base + quote(filename)
        try:
            content = _fetch(url)
        except OSError as exc:
            errors.append(f"{url} : {exc}")
            continue

        digest = _sha256(content)
        if digest != expected:
            errors.append(f"{url} : empreinte inattendue ({digest})")
            continue

        target.write_bytes(content)
        print(f"{filename} : téléchargé et vérifié ({len(content) / 1e6:.1f} Mo)")
        return

    raise RuntimeError(f"Impossible de télécharger {filename} :\n" + "\n".join(errors))


def download_dataset(dest_dir=DEFAULT_DATA_DIR, force=False):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected in EXPECTED_SHA256.items():
        target = dest_dir / filename
        if target.exists() and not force and _sha256(target.read_bytes()) == expected:
            print(f"{filename} : déjà présent et vérifié")
            continue
        _download_file(filename, expected, target)


def main():
    try:
        download_dataset()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
