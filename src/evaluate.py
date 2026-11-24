"""
src/evaluate.py
---------------
Evaluation suite for Roman-script Hinglish Sentiment Analysis:
- Metrics: Macro-F1 (primary), Accuracy, Weighted-F1, Per-class Precision/Recall/F1
- Confusion Matrix plotting and saving
- Linguistic Error Analysis: categorizes errors by linguistic cause:
  (sarcasm, Hindi negation, heavy code-switching, spelling variation, neutral ambiguity)
- Multi-seed aggregation (mean ± std)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

try:
    from src.data import ID2LABEL, LABEL2ID
except ImportError:
    from data import ID2LABEL, LABEL2ID

CLASS_NAMES = [ID2LABEL[i] for i in range(len(ID2LABEL))]

# Hindi negation markers frequently found in Hinglish
NEGATION_WORDS = {"nahi", "nahin", "nhi", "mat", "kabhi nahi", "na", "bilkul nahi", "nope"}
POSITIVE_LEXICON = {"zabardast", "mast", "superb", "shandar", "dhanyawad", "badhiya", "best", "love", "amazing", "proud"}


def compute_metrics_for_trainer(eval_pred) -> Dict[str, float]:
    """
    Evaluation callback for Hugging Face Trainer.
    Calculates Macro-F1 (primary metric), Accuracy, and Weighted-F1.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "weighted_f1": float(weighted_f1)
    }


def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation returning overall and per-class metrics.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": float(precisions[idx]),
            "recall": float(recalls[idx]),
            "f1": float(f1s[idx])
        }

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": conf_mat
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None
):
    """
    Renders and saves a publication-quality confusion matrix heatmap.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=11)
    plt.ylabel("True Sentiment", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved confusion matrix to: {save_path}")


def categorize_error_reason(text: str, true_label: str, pred_label: str) -> str:
    """
    Heuristically groups misclassified Hinglish tweets into root causes:
    - Sarcasm / Irony
    - Hindi Negation Confusion
    - Spelling Variation / Elongation
    - Neutral / Subtle Ambiguity
    - Heavy Code-Switching
    """
    lower = text.lower()
    tokens = set(re.findall(r'\b\w+\b', lower))

    # 1. Hindi Negation check (e.g. 'nahi', 'mat')
    if tokens.intersection(NEGATION_WORDS):
        if true_label == "negative" and pred_label in ["positive", "neutral"]:
            return "Hindi Negation (Missed Polarity Flip)"

    # 2. Sarcasm detection heuristic (presence of positive marker in negative context or sarcastic clap/emoji)
    if "👏" in text or "😂" in text or "dhanyawad" in lower or "shukriya" in lower:
        if true_label == "negative" and pred_label == "positive":
            return "Sarcasm / Mock Praise"

    # 3. Repeated character elongation (spelling variation)
    if re.search(r'(.)\1{2,}', text):
        return "Spelling Variation / Character Elongation"

    # 4. Neutral Confusion (Ambiguity / Factual vs Opinion)
    if true_label == "neutral" or pred_label == "neutral":
        return "Neutral-Sentiment Ambiguity"

    # 5. Heavy Code-Switching default
    return "Complex Code-Switching / Mixed Sentiment"


def perform_error_analysis(
    texts: List[str],
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    output_csv: Optional[str] = None,
    max_samples: int = 50
) -> pd.DataFrame:
    """
    Extracts misclassified instances and categorizes them by linguistic cause.
    """
    records = []
    for idx, (t, yt, yp) in enumerate(zip(texts, y_true, y_pred)):
        if yt != yp:
            true_name = ID2LABEL[yt]
            pred_name = ID2LABEL[yp]
            cause = categorize_error_reason(t, true_name, pred_name)
            
            conf = float(np.max(y_probs[idx])) if y_probs is not None else 0.0

            records.append({
                "index": idx,
                "text": t,
                "true_label": true_name,
                "predicted_label": pred_name,
                "confidence": round(conf, 4),
                "linguistic_cause": cause
            })

    err_df = pd.DataFrame(records)
    if len(err_df) == 0:
        return err_df

    groups = [g for _, g in err_df.groupby("linguistic_cause")]
    picked = []
    cursors = [0] * len(groups)
    while len(picked) < max_samples:
        progressed = False
        for i, group in enumerate(groups):
            if cursors[i] < len(group):
                picked.append(group.iloc[cursors[i]])
                cursors[i] += 1
                progressed = True
                if len(picked) >= max_samples:
                    break
        if not progressed:
            break
    sample = pd.DataFrame(picked)
    if output_csv:
        parent = os.path.dirname(output_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)
        sample.to_csv(output_csv, index=False)
        counts = err_df["linguistic_cause"].value_counts().rename_axis("linguistic_cause").reset_index(name="count")
        counts_path = os.path.splitext(output_csv)[0] + "_counts.csv"
        counts.to_csv(counts_path, index=False)
        print(f"[Analysis] Saved {len(sample)} of {len(err_df)} errors to: {output_csv}")
    return sample


def aggregate_seed_results(seed_results: List[Dict[str, float]]) -> Dict[str, str]:
    """
    Given a list of metric dicts across multiple random seeds,
    computes and formats `mean ± std`.
    """
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in seed_results if k in r]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            summary[f"{k}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
            summary[f"{k}_mean"] = f"{mean:.4f}"
            summary[f"{k}_std"] = f"{std:.4f}"
    return summary


def plot_training_curves(history: List[dict], save_path: str, title: str):
    """Plot train loss and validation macro-F1 from a Trainer log history."""
    train_epochs, train_loss = [], []
    val_epochs, val_f1 = [], []
    for row in history:
        if "loss" in row and "epoch" in row:
            train_epochs.append(row["epoch"])
            train_loss.append(row["loss"])
        if "eval_macro_f1" in row:
            val_epochs.append(row.get("epoch", len(val_epochs) + 1))
            val_f1.append(row["eval_macro_f1"])
    if not train_loss and not val_f1:
        return
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if train_loss:
        marker = "o" if len(train_loss) <= 20 else None
        axes[0].plot(train_epochs, train_loss, marker=marker, color="#1a5276")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    if val_f1:
        axes[1].plot(val_epochs, val_f1, marker="o", color="#196f3d")
    axes[1].set_title("Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[Plot] {save_path}")


def plot_model_comparison(summary_df: pd.DataFrame, save_path: str):
    """Bar chart of macro-F1 with standard-deviation error bars."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = summary_df.sort_values("macro_f1_mean")
    ax.barh(
        order["model"],
        order["macro_f1_mean"],
        xerr=order["macro_f1_std"].fillna(0),
        color="#1a5276",
        capsize=3,
    )
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1)
    ax.set_title("Test macro-F1, mean over seeds")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[Plot] {save_path}")


if __name__ == "__main__":
    # Test sample evaluation
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1]
    res = evaluate_predictions(y_true, y_pred)
    print("Test Accuracy:", res["accuracy"])
    print("Test Macro-F1:", res["macro_f1"])
    print("Per-class metrics:", res["per_class"])
