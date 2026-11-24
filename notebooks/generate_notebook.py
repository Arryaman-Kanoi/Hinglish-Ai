"""Rebuild notebooks/colab_sentimix_hinglish.ipynb from the src package.

The notebook is the run. Each code cell is a step of the project, and its
printout and plots stay in the file after it executes.
"""

import json
import os
import uuid

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python (sentimix)", "language": "python", "name": "sentimix"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [],
}


def md(source):
    nb["cells"].append({
        "id": uuid.uuid4().hex[:8],
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    })


def code(source):
    nb["cells"].append({
        "id": uuid.uuid4().hex[:8],
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })


md("""# SentiMix Hinglish sentiment

**CS3255 Natural Language Processing — SEE project**
**Student:** Krishnendu Nayak &nbsp; **USN:** 1RUA24SCS0052

Three-class sentiment on Roman Hinglish: negative, neutral, positive. The data is the official SemEval-2020 Task 9 Hinglish split. Macro-F1 is the main number. Accuracy is reported beside it.

This notebook is the run. Each cell is one step. The printout under the cell is what ran. Running it again does not start from scratch: a model seed already stored in `results/metrics_seeds.csv` prints `[Skip]`.

The hard slice is a flip: the obvious word says one attitude, and a negation or a contrast word reverses it. Those extra training rows are added in the last training cell. The official test tweets are never edited, and they are not used to pick the learning rate.
""")

md("## 1. Setup")

code("""import os, sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Colab only. The local environment already has the CUDA build of torch.
if "google.colab" in sys.modules:
    !pip install -q transformers datasets accelerate scikit-learn emoji matplotlib seaborn pandas

print("Project root:", ROOT.resolve())
""")

md("""## 2. Load the official split

Train and validation lose duplicate text and any text that also appears in a later split. The released test file stays as it was published.
""")

code("""import json
import pandas as pd

from src.data import load_official_sentimix, load_transfer_csv
from src.eda import run_eda
from src.experiments import (
    RESULTS,
    ROOT,
    TRANSFER_CSV,
    attach_clean,
    choose_max_len,
    hardware_config,
)

print("[Data] Loading official SentiMix Hinglish split")
train_df, val_df, test_df = load_official_sentimix(os.path.join(ROOT, "data", "raw"))
for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
    keep = [c for c in ("uid", "text", "sentiment", "label", "label_text", "cmi", "n_tokens", "n_hin", "n_eng") if c in frame.columns]
    frame[keep].to_csv(os.path.join(ROOT, "data", f"sentimix_{name}.csv"), index=False)

run_eda({"train": train_df, "validation": val_df, "test": test_df}, RESULTS)
train_df = attach_clean(train_df)
val_df = attach_clean(val_df)
test_df = attach_clean(test_df)

print("rows", len(train_df), len(val_df), len(test_df))
print(train_df["label_text"].value_counts().to_string())
""")

md("""## 3. Length and hardware

Sequence length is the 95th percentile of mBERT subword lengths, rounded up to a multiple of 16, and capped for a 4GB GPU.
""")

code("""hw = hardware_config()
max_len, p95, truncated = choose_max_len(
    pd.concat([train_df["clean_text"], val_df["clean_text"], test_df["clean_text"]]),
    hw["max_len_cap"],
)
config = {
    "seeds": [42, 100, 2024],
    "seeds_note": "HingBERT uses three seeds. mBERT, MuRIL, IndicBERTv2, and XLM-RoBERTa use seed 42.",
    "max_len": max_len,
    "subword_p95_mbert": p95,
    "fraction_over_max_len_in_sample": truncated,
    "hardware": hw,
    "official_split": "SemEval-2020 Task 9 Hinglish, Zenodo 3974927",
    "test_used_for": "final evaluation only; learning rate chosen on validation",
}
os.makedirs(RESULTS, exist_ok=True)
with open(os.path.join(RESULTS, "training_config.json"), "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
print(f"max_len={max_len}  subword p95={p95:.1f}  truncated={truncated:.3f}")
print(hw)
""")

md("""## 4. TF-IDF baselines

Logistic regression and a linear SVM. Three seeds. A seed already in `results/metrics_seeds.csv` prints `[Skip]` and is not trained again.
""")

code("""from src.experiments import run_linear_models

run_linear_models(train_df, val_df, test_df)
""")

md("## 5. BiLSTM baseline")

code("""from src.experiments import run_bilstm_seeds, write_summary

run_bilstm_seeds(train_df, val_df, test_df)
write_summary()
""")

md("""## 6. Learning rate

HingBERT at 2e-5, 3e-5, and 5e-5. The choice uses validation macro-F1 only.
""")

code("""from src.experiments import select_learning_rate

learning_rate = select_learning_rate(train_df, val_df, test_df, max_len, hw)
config["learning_rate"] = learning_rate
with open(os.path.join(RESULTS, "training_config.json"), "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
print("selected learning rate", learning_rate)
""")

md("""## 7. HingBERT

Three seeds, at the learning rate chosen above. A finished seed is skipped.
""")

code("""from src.experiments import run_one_transformer

run_one_transformer("hingbert", train_df, val_df, test_df, max_len, hw, learning_rate)
""")

md("## 8. mBERT")

code("""from src.experiments import run_one_transformer

run_one_transformer("mbert", train_df, val_df, test_df, max_len, hw, learning_rate)
""")

md("## 9. MuRIL")

code("""from src.experiments import run_one_transformer

run_one_transformer("muril", train_df, val_df, test_df, max_len, hw, learning_rate)
""")

md("## 10. IndicBERTv2")

code("""from src.experiments import run_one_transformer

run_one_transformer("indicbert", train_df, val_df, test_df, max_len, hw, learning_rate)
""")

md("## 11. XLM-RoBERTa")

code("""from src.experiments import run_one_transformer

run_one_transformer("xlm-roberta", train_df, val_df, test_df, max_len, hw, learning_rate)
""")

md("""## 12. Ablations

HingBERT, seed 42. Raw text, no emoji, no class weights, frozen encoder. One that is already in `results/ablations.csv` is skipped.
""")

code("""from src.experiments import run_ablations, write_error_report, write_summary

run_ablations(train_df, val_df, test_df, max_len, hw, learning_rate)
write_summary()
write_error_report()
""")

md("""## 13. Second dataset

The best checkpoint is scored on the other Roman Hinglish file. This cell does not train.
""")

code("""from src.experiments import attach_clean as clean_text
from src.experiments import run_transfer

if os.path.exists(TRANSFER_CSV):
    blocked = pd.concat([train_df["text"], val_df["text"], test_df["text"]]).tolist()
    transfer = load_transfer_csv(TRANSFER_CSV, blocked_texts=blocked)
    transfer = clean_text(transfer)
    transfer.to_csv(os.path.join(ROOT, "data", "transfer_hinglish.csv"), index=False)
    run_transfer(transfer)
else:
    print("No second dataset at", TRANSFER_CSV)
""")

md("""## 14. Flip epoch

One more HingBERT epoch on the training tweets plus the flip rows. Kept only if validation allows it. The test set is scored after that decision. If `results/flip_model_metrics.csv` already exists, this cell skips.
""")

code("""from src.experiments import _guard, run_flip_hingbert

_guard(
    "hingbert flip",
    lambda: run_flip_hingbert(train_df, val_df, test_df, max_len, hw, learning_rate),
)
print("[Done] Results are in", RESULTS)
""")

md("""## 15. Test table

Macro-F1 is the mean across negative, neutral, and positive. The ± value is the sample standard deviation across seeds. One-seed models have an empty standard deviation.
""")

code("""from IPython.display import display, Markdown

metrics = pd.read_csv(os.path.join(RESULTS, "metrics.csv"))
show = metrics.copy()
for col in ("accuracy", "macro_f1", "weighted_f1"):
    mean, std = f"{col}_mean", f"{col}_std"
    show[col] = show.apply(
        lambda r, mean=mean, std=std: (
            f"{r[mean]:.4f}" if pd.isna(r[std]) else f"{r[mean]:.4f} ± {r[std]:.4f}"
        ),
        axis=1,
    )
keep = ["model", "accuracy", "macro_f1", "weighted_f1", "train_seconds_mean", "params", "n_seeds"]
display(show[keep].sort_values("macro_f1", ascending=False))

for name in ("lr_grid.csv", "ablations.csv", "transfer.csv", "flip_model_metrics.csv", "flip_added/test.csv", "slices.csv"):
    path = Path(RESULTS) / name
    if path.exists():
        display(Markdown(f"### {name}"))
        display(pd.read_csv(path))
""")

md("## 16. Graphs")

code("""from IPython.display import display, Image, Markdown

plot_dir = Path(RESULTS) / "plots"
names = [
    "class_distribution.png",
    "length_histogram.png",
    "cmi_distribution.png",
    "top_words.png",
    "model_comparison.png",
]
names += sorted(p.name for p in plot_dir.glob("cm_*.png"))
names += sorted(p.name for p in plot_dir.glob("curves_*.png"))
seen = set()
for name in names:
    if name in seen:
        continue
    seen.add(name)
    path = plot_dir / name
    if path.exists():
        display(Markdown(f"**{name}**"))
        display(Image(filename=str(path)))
""")

md("## 17. Errors")

code("""from IPython.display import display

counts = Path(RESULTS) / "errors_best_counts.csv"
errors = Path(RESULTS) / "errors_best.csv"
if counts.exists():
    display(pd.read_csv(counts))
if errors.exists():
    pd.set_option("display.max_colwidth", 160)
    display(pd.read_csv(errors).head(20))
else:
    print("errors_best.csv is written after the ablation cell.")
""")

md("## 18. Try a few lines")

code("""from src.predict import predict

samples = [
    "Bhai yeh movie sach me ekdum zabardast hai! 🔥",
    "Delivery itni late thi, bilkul bekaar service 😡",
    "Match kal shaam 7:30 baje start hoga.",
    "Ye phone lene ki galti mat karna, battery nahi chalti.",
    "Me acha nahi hu na",
    "Package deliver ho gaya hai but maine abhi open nahi kiya.",
]
for text in samples:
    out = predict(text)
    print(f"{out['label']:8}  {out['confidence']:.2f}  {out.get('flip', '')}")
    print(" ", text)
    print(" ", out.get("reading", ""))
""")

out = os.path.join(os.path.dirname(__file__), "colab_sentimix_hinglish.ipynb")
with open(out, "w", encoding="utf-8") as handle:
    json.dump(nb, handle, indent=2)
print("wrote", out, "cells", len(nb["cells"]))
