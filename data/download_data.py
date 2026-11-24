"""Download the SentiMix Hinglish release if the CoNLL files are not already local.

Source: Zenodo record 3974927, which includes train, validation, and test labels.
The second file is the Roman-script Hinglish set at DOI 10.17632/wtp7htm3ss.1.
"""

import os
import zipfile
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
ZENODO_URL = "https://zenodo.org/records/3974927/files/Semeval_2020_task9_data.zip?download=1"
MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/wtp7htm3ss/files/"
    "dafb5ce5-3c10-4a6b-8881-5d65312cfd80/file_downloaded"
)


def _fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[Download] Already present: {dest}")
        return dest
    print(f"[Download] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def ensure_sentimix():
    hing = os.path.join(ROOT, "raw", "sentimix", "Semeval_2020_task9_data", "Hinglish")
    train = os.path.join(hing, "Hinglish_train_14k_split_conll.txt")
    if os.path.exists(train):
        print(f"[Download] SentiMix CoNLL already at {hing}")
        return hing
    zip_path = os.path.join(ROOT, "raw", "Semeval_2020_task9_data.zip")
    _fetch(ZENODO_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(os.path.join(ROOT, "raw", "sentimix"))
    return hing


def ensure_transfer():
    dest = os.path.join(ROOT, "raw", "second", "hinglish_social.csv")
    return _fetch(MENDELEY_URL, dest)


if __name__ == "__main__":
    ensure_sentimix()
    ensure_transfer()
