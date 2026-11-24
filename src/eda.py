"""Dataset description plots and tables for the SentiMix Hinglish splits."""

import json
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import emoji as emoji_lib
except ImportError:
    emoji_lib = None

CLASS_ORDER = ["negative", "neutral", "positive"]
CLASS_COLORS = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#1e8449"}

# Closed-class words dropped only from the top-word chart. Not used in training.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "it", "this",
    "that", "with", "as", "at", "be", "by", "from", "are", "was", "were", "i", "you",
    "we", "he", "she", "they", "my", "your", "our", "not", "but", "so", "if", "rt",
    "hai", "hain", "ho", "tha", "thi", "the", "ka", "ki", "ke", "ko", "se", "me",
    "mein", "par", "pe", "aur", "ya", "toh", "to", "hi", "bhi", "ne", "ek", "ye",
    "yeh", "woh", "wo", "bhi", "se", "ka", "ki",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def _has_emoji(text: str) -> bool:
    if emoji_lib is not None:
        return emoji_lib.emoji_count(text) > 0
    return EMOJI_RE.search(text) is not None


def describe_frame(df: pd.DataFrame, split_name: str) -> dict:
    text = df["text"].astype(str)
    tokens = df["n_tokens"] if "n_tokens" in df.columns else text.str.split().str.len()
    dev_chars = text.map(lambda s: len(DEVANAGARI_RE.findall(s)))
    latin_chars = text.map(lambda s: len(re.findall(r"[A-Za-z]", s)))
    stats = {
        "split": split_name,
        "n": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["label_text"].value_counts().items()},
        "avg_tokens": float(tokens.mean()),
        "median_tokens": float(tokens.median()),
        "p95_tokens": float(tokens.quantile(0.95)),
        "avg_chars": float(text.str.len().mean()),
        "devanagari_char_share": float(dev_chars.sum() / max(1, (dev_chars + latin_chars).sum())),
        "emoji_tweet_share": float(text.map(_has_emoji).mean()),
        "url_tweet_share": float(text.str.contains(r"https?://|www\.", case=False, regex=True).mean()),
        "mention_tweet_share": float(text.str.contains(r"@\w+", regex=True).mean()),
    }
    if "cmi" in df.columns:
        stats["cmi_mean"] = float(df["cmi"].mean())
        stats["cmi_median"] = float(df["cmi"].median())
        stats["cmi_pct_mixed"] = float((df["cmi"] > 0).mean())
    return stats


def _top_words(df: pd.DataFrame, n: int = 12) -> dict:
    per_class = {}
    for name in CLASS_ORDER:
        counts = Counter()
        subset = df.loc[df["label_text"] == name, "text"].astype(str)
        for text in subset:
            for tok in TOKEN_RE.findall(text.lower()):
                if tok in STOPWORDS or len(tok) < 3:
                    continue
                counts[tok] += 1
        per_class[name] = counts.most_common(n)
    return per_class


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[Plot] {path}")


def run_eda(splits: dict, out_dir: str) -> dict:
    """Write dataset stats and the EDA plots. `splits` maps name -> DataFrame."""
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    all_df = pd.concat(
        [df.assign(split=name) for name, df in splits.items()],
        ignore_index=True,
    )
    stats = {name: describe_frame(df, name) for name, df in splits.items()}
    stats["all"] = describe_frame(all_df, "all")
    stats["top_words"] = _top_words(all_df)

    stats_path = os.path.join(out_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"[EDA] Wrote {stats_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    count_df = (
        all_df.groupby(["split", "label_text"]).size().rename("count").reset_index()
    )
    sns.barplot(
        data=count_df,
        x="split",
        y="count",
        hue="label_text",
        hue_order=CLASS_ORDER,
        palette=CLASS_COLORS,
        ax=ax,
    )
    ax.set_title("Class distribution by official split")
    ax.set_xlabel("")
    ax.set_ylabel("Tweets")
    ax.legend(title="")
    _save(fig, os.path.join(plot_dir, "class_distribution.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    token_col = "n_tokens" if "n_tokens" in all_df.columns else None
    lengths = all_df[token_col] if token_col else all_df["text"].astype(str).str.split().str.len()
    sns.histplot(lengths, bins=40, ax=ax, color="#2471a3")
    p95 = float(np.quantile(lengths, 0.95))
    ax.axvline(p95, color="#c0392b", linestyle="--", label=f"95th percentile ({p95:.0f})")
    ax.set_title("Tweet length in CoNLL tokens")
    ax.set_xlabel("Tokens")
    ax.legend()
    _save(fig, os.path.join(plot_dir, "length_histogram.png"))

    if "cmi" in all_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(all_df["cmi"], bins=30, ax=ax, color="#6c3483")
        ax.set_title("Code-mixing index (gold Hin/Eng tags)")
        ax.set_xlabel("CMI")
        _save(fig, os.path.join(plot_dir, "cmi_distribution.png"))

    top = stats["top_words"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, name in zip(axes, CLASS_ORDER):
        words, counts = zip(*top[name]) if top[name] else ([], [])
        ax.barh(list(words)[::-1], list(counts)[::-1], color=CLASS_COLORS[name])
        ax.set_title(name)
        ax.set_xlabel("Count")
    fig.suptitle("Most frequent content words")
    _save(fig, os.path.join(plot_dir, "top_words.png"))
    return stats
