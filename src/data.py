"""
src/data.py
-----------
Dataset loading, deduplication, stratified splitting, and PyTorch/HuggingFace dataset wrappers
for SentiMix Roman-script Hinglish Sentiment Analysis.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset

# Canonical label mapping (Standardized across entire repository)
LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Flexible synonyms for label mapping
LABEL_SYNONYMS = {
    "negative": 0, "neg": 0, "-1": 0, -1: 0, 0: 0,
    "neutral": 1, "neu": 1, "0": 1, 1: 1,
    "positive": 2, "pos": 2, "1": 2, "+1": 2, 2: 2
}


def normalize_label(val) -> int:
    """Normalizes various label representations to {0: neg, 1: neu, 2: pos}."""
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in LABEL_SYNONYMS:
            return LABEL_SYNONYMS[val_clean]
    elif val in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[val]
    raise ValueError(f"Unrecognized sentiment label: {val}")


def deduplicate_dataset(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Deduplicates tweets to prevent data leakage across train/val/test splits.
    Also strips retweet prefix 'RT @USER:' to identify semantic duplicates.
    """
    initial_len = len(df)
    
    # Create temporary normalized text for deduplication check
    norm_text = (
        df[text_col]
        .astype(str)
        .str.replace(r'^RT\s+@\w+:\s*', '', regex=True)
        .str.strip()
        .str.lower()
    )
    
    mask = ~norm_text.duplicated(keep="first")
    df_dedup = df[mask].copy().reset_index(drop=True)
    
    dropped = initial_len - len(df_dedup)
    if dropped > 0:
        print(f"[Data] Deduplication: Removed {dropped} duplicate/retweet records ({initial_len} -> {len(df_dedup)}).")
    return df_dedup


def load_dataset_from_file(
    filepath: str,
    text_col: str = "text",
    label_col: str = "sentiment",
    dedup: bool = True
) -> pd.DataFrame:
    """
    Loads dataset from CSV or TSV, validates columns, maps labels, and optionally deduplicates.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    sep = "\t" if filepath.endswith(".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)

    # Rename common column variations
    col_mapping = {
        "tweet": "text", "Tweet": "text", "sentence": "text", "Text": "text",
        "label": "sentiment", "Sentiment": "sentiment", "class": "sentiment", "Label": "sentiment"
    }
    df = df.rename(columns=col_mapping)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset must contain '{text_col}' and '{label_col}' columns. Found: {list(df.columns)}")

    # Clean missing values
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)

    # Map labels to 0, 1, 2
    df["label"] = df[label_col].apply(normalize_label)
    df["label_text"] = df["label"].map(ID2LABEL)

    if dedup:
        df = deduplicate_dataset(df, text_col=text_col)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_col: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified 3-way split (Train / Validation / Test) using fixed seed.
    Guarantees no data leakage and balanced class representation.
    """
    # 1. Split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col]
    )

    # 2. Split train_val into train and validation
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_df[stratify_col]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights_tensor(labels: List[int], device: str = "cpu") -> torch.Tensor:
    """
    Computes balanced class weights for Cross-Entropy loss:
        weight[c] = total_samples / (n_classes * count[c])
    """
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    # Cross-entropy wants one weight per label id, even if a class is absent.
    full = np.ones(len(LABEL2ID), dtype=np.float32)
    for cls, weight in zip(present, weights):
        full[int(cls)] = float(weight)
    return torch.tensor(full, dtype=torch.float32, device=device)


class HinglishSentimentDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized inputs and class labels.
    """
    def __init__(self, encodings: Dict[str, List], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values())))


# Gold language tags in the SentiMix CoNLL release.
_LANG_DEPENDENT = {"Hin", "Eng"}
_URL_START = {"http", "https"}
_URL_STRUCT = {"//", "/", ":", ".", "t", "co", "tco", "www"}


def code_mix_index(tags: List[str]) -> float:
    """Gambäck and Das (2014) CMI from gold Hin/Eng tags.

    Language-independent tags (O, EMT, punctuation) are excluded.
    0 means one language or no language-tagged tokens.
    """
    counts: Dict[str, int] = {}
    for tag in tags:
        if tag in _LANG_DEPENDENT:
            counts[tag] = counts.get(tag, 0) + 1
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return 100.0 * (1.0 - (max(counts.values()) / n))


def _take_url(tokens: List[str], start: int) -> Tuple[str, int]:
    """Join a CoNLL-split URL back into one token. Returns (url, next_index)."""
    parts: List[str] = []
    j = start
    while j < len(tokens) and len(parts) < 12:
        tok = tokens[j]
        low = tok.lower()
        if not parts:
            if low not in _URL_START:
                break
            parts.append(low)
            j += 1
            continue
        if tok in {"//", "/", ":", "."} or low in _URL_STRUCT:
            parts.append(tok if tok in {"//", "/", ":", "."} else low)
            j += 1
            continue
        if parts[-1] == "/" and re.fullmatch(r"[A-Za-z0-9_-]+", tok):
            parts.append(tok)
            j += 1
            break
        if "/" not in parts and re.fullmatch(r"[A-Za-z0-9]+", tok) and len(tok) <= 16:
            parts.append(low)
            j += 1
            continue
        break
    raw = "".join(parts)
    raw = raw.replace("https//", "https://").replace("http//", "http://")
    raw = raw.replace("https:/", "https://").replace("http:/", "http://")
    raw = re.sub(r"(https?://)/+", r"\1", raw)
    # The CoNLL release writes "t.co" as the single token "tco".
    raw = raw.replace("://tco/", "://t.co/")
    if raw in {"http", "https", "http://", "https://"}:
        return tokens[start], start + 1
    return raw, j


def reconstruct_tweet(tokens: List[str]) -> str:
    """Rebuild a tweet from CoNLL tokens.

    '@' + handle becomes @handle, '#' + word becomes #word, and split
    t.co links become a single URL. Other tokens stay space-separated.
    """
    words: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        low = tok.lower()
        if low in _URL_START:
            url, i = _take_url(tokens, i)
            words.append(url)
            continue
        if tok == "@" and i + 1 < n and re.fullmatch(r"[A-Za-z0-9_]+", tokens[i + 1]):
            words.append("@" + tokens[i + 1])
            i += 2
            continue
        if tok == "#" and i + 1 < n and re.fullmatch(r"\w+", tokens[i + 1], flags=re.UNICODE):
            words.append("#" + tokens[i + 1])
            i += 2
            continue
        words.append(tok)
        i += 1
    text = " ".join(words)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_sentimix_conll(path: str, labels_csv: Optional[str] = None) -> pd.DataFrame:
    """Parse a SentiMix Hinglish CoNLL file into one row per tweet.

    Train and validation store the label on the meta line. The test file does
    not, so pass labels_csv (Uid, Sentiment) for that split.
    """
    label_by_uid: Dict[str, str] = {}
    if labels_csv:
        lab = pd.read_csv(labels_csv)
        uid_col = "Uid" if "Uid" in lab.columns else lab.columns[0]
        sent_col = "Sentiment" if "Sentiment" in lab.columns else lab.columns[1]
        label_by_uid = {str(u): str(s).strip().lower() for u, s in zip(lab[uid_col], lab[sent_col])}

    records = []
    tokens: List[str] = []
    tags: List[str] = []
    uid: Optional[str] = None
    sentiment: Optional[str] = None

    def flush():
        nonlocal tokens, tags, uid, sentiment
        if uid is None:
            return
        label = sentiment
        if label is None:
            label = label_by_uid.get(str(uid))
        if label is None or not tokens:
            tokens, tags, uid, sentiment = [], [], None, None
            return
        records.append(
            {
                "uid": str(uid),
                "text": reconstruct_tweet(tokens),
                "sentiment": str(label).strip().lower(),
                "cmi": code_mix_index(tags),
                "n_tokens": len(tokens),
                "n_hin": sum(1 for t in tags if t == "Hin"),
                "n_eng": sum(1 for t in tags if t == "Eng"),
                "n_emoji_tag": sum(1 for t in tags if t == "EMT"),
            }
        )
        tokens, tags, uid, sentiment = [], [], None, None

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if not raw.strip():
                flush()
                continue
            parts = raw.split("\t")
            if parts[0] == "meta":
                flush()
                uid = parts[1].strip()
                sentiment = parts[2].strip().lower() if len(parts) > 2 and parts[2].strip() else None
                tokens, tags = [], []
                continue
            tokens.append(parts[0])
            tags.append(parts[1] if len(parts) > 1 else "O")
        flush()

    df = pd.DataFrame(records)
    df["label"] = df["sentiment"].apply(normalize_label)
    df["label_text"] = df["label"].map(ID2LABEL)
    return df


def _norm_key(text: str) -> str:
    text = re.sub(r"^RT\s+", "", str(text), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def drop_cross_split_leaks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop duplicates inside each split and any train/val text that also appears in test.

    The official test set is left unchanged. Leaking copies are removed from
    train, then from validation.
    """
    def dedup_keep(df: pd.DataFrame) -> pd.DataFrame:
        keys = df[text_col].map(_norm_key)
        return df.loc[~keys.duplicated(keep="first")].reset_index(drop=True)

    # The official test file stays intact, including any repeated texts.
    test_dups = int(test_df[text_col].map(_norm_key).duplicated().sum())
    val_df = dedup_keep(val_df)
    train_df = dedup_keep(train_df)

    test_keys = set(test_df[text_col].map(_norm_key))
    val_keys = set(val_df[text_col].map(_norm_key))

    before_train = len(train_df)
    train_df = train_df.loc[~train_df[text_col].map(_norm_key).isin(test_keys | val_keys)].reset_index(drop=True)
    before_val = len(val_df)
    val_df = val_df.loc[~val_df[text_col].map(_norm_key).isin(test_keys)].reset_index(drop=True)
    print(
        f"[Data] Leakage filter: train {before_train} -> {len(train_df)}, "
        f"val {before_val} -> {len(val_df)}, test {len(test_df)} left as released "
        f"({test_dups} duplicate texts kept)."
    )
    return train_df, val_df, test_df


def sentimix_paths(raw_root: str) -> Dict[str, str]:
    hing = os.path.join(raw_root, "Semeval_2020_task9_data", "Hinglish")
    if not os.path.isdir(hing):
        hing = os.path.join(raw_root, "sentimix", "Semeval_2020_task9_data", "Hinglish")
    test_name = "Hinglish_test_unalbelled_conll_updated.txt"
    return {
        "train": os.path.join(hing, "Hinglish_train_14k_split_conll.txt"),
        "val": os.path.join(hing, "Hinglish_dev_3k_split_conll.txt"),
        "test": os.path.join(hing, test_name),
        "test_labels": os.path.join(hing, "Hinglish_test_labels.txt"),
    }


def load_official_sentimix(raw_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the official SentiMix Hinglish train / validation / test split."""
    paths = sentimix_paths(raw_root)
    for key in ("train", "val", "test", "test_labels"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"Missing SentiMix file: {paths[key]}")
    train_df = parse_sentimix_conll(paths["train"])
    val_df = parse_sentimix_conll(paths["val"])
    test_df = parse_sentimix_conll(paths["test"], labels_csv=paths["test_labels"])
    print(f"[Data] Parsed train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return drop_cross_split_leaks(train_df, val_df, test_df)


def load_transfer_csv(path: str, blocked_texts: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a second labeled set and drop texts that also occur in SentiMix."""
    df = pd.read_csv(path)
    df = df.rename(columns={"Tweet": "text", "Text": "text", "Sentence": "text"})
    if "text" not in df.columns:
        raise ValueError(f"Transfer file needs a text column. Found {list(df.columns)}")
    if "sentiment" in df.columns:
        df["label"] = df["sentiment"].apply(normalize_label)
    elif "label" in df.columns:
        df["label"] = df["label"].apply(normalize_label)
    else:
        raise ValueError("Transfer file needs a sentiment or label column.")
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.dropna(subset=["text"]).copy()
    df = df.loc[df["text"].str.len() > 0].copy()
    df["label_text"] = df["label"].map(ID2LABEL)
    df["sentiment"] = df["label_text"]
    keys = df["text"].map(_norm_key)
    df = df.loc[~keys.duplicated(keep="first")].copy()
    if blocked_texts:
        blocked = {_norm_key(t) for t in blocked_texts}
        before = len(df)
        df = df.loc[~df["text"].map(_norm_key).isin(blocked)].copy()
        print(f"[Data] Transfer overlap with SentiMix removed: {before} -> {len(df)}")
    df = df.reset_index(drop=True)
    return df[["text", "sentiment", "label", "label_text"]]


if __name__ == "__main__":
    sample_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample_hinglish.csv")
    if os.path.exists(sample_csv):
        df = load_dataset_from_file(sample_csv)
        print("Loaded DataFrame shape:", df.shape)
        train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1, random_state=42)
        print("Class counts in Train:\n", train_df['label_text'].value_counts())
        weights = compute_class_weights_tensor(train_df['label'].tolist())
        print("Computed class weights tensor:", weights)
