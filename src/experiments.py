"""Run the SentiMix experiments and write results/ as they finish.

The official train, validation, and test files stay fixed. Three seeds are
three initializations. The learning-rate choice uses validation macro-F1 only.
"""

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_official_sentimix, load_transfer_csv
from src.eda import run_eda
from src.evaluate import (
    plot_model_comparison,
    plot_training_curves,
    perform_error_analysis,
)
from src.models import MODEL_REGISTRY
from src.preprocess import preprocess_corpus
from src.train import (
    parameter_count,
    release_memory,
    train_baseline,
    train_bilstm,
    train_transformer_once,
    save_checkpoint,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEEDS = [42, 100, 2024]
TRANSFORMERS = ["hingbert", "mbert", "muril", "indicbert", "xlm-roberta"]
LEARNING_RATES = [2e-5, 3e-5, 5e-5]
TRANSFER_CSV = os.path.join(ROOT, "data", "raw", "second", "hinglish_social.csv")


def _read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _append_row(path, row):
    frame = _read_csv(path)
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _done_pairs(path):
    frame = _read_csv(path)
    if frame.empty or "model" not in frame.columns:
        return set()
    return set(zip(frame["model"].astype(str), frame["seed"].astype(int)))


def attach_clean(frame, strip_emojis=False, raw=False):
    out = frame.copy()
    if raw:
        out["clean_text"] = out["text"].astype(str)
    else:
        out["clean_text"] = preprocess_corpus(out["text"].tolist(), strip_emojis=strip_emojis)
    return out


def hardware_config():
    if not torch.cuda.is_available():
        return {
            "batch_size": 4,
            "grad_accum": 4,
            "fp16": False,
            "optim_name": "adamw_torch",
            "max_len_cap": 64,
            "device": "cpu",
        }
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    optim_name = "adamw_torch"
    if vram < 8:
        try:
            import bitsandbytes  # noqa: F401
            optim_name = "adamw_bnb_8bit"
        except ImportError:
            optim_name = "adafactor"
        return {
            "batch_size": 2,
            "grad_accum": 8,
            "fp16": True,
            "optim_name": optim_name,
            "max_len_cap": 96,
            "device": name,
            "vram_gb": round(vram, 2),
        }
    return {
        "batch_size": 8 if vram < 14 else 16,
        "grad_accum": 2 if vram < 14 else 1,
        "fp16": True,
        "optim_name": "adamw_torch",
        "max_len_cap": 128,
        "device": name,
        "vram_gb": round(vram, 2),
    }


def choose_max_len(texts, cap):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    step = max(1, len(texts) // 2500)
    sample = list(texts)[::step][:2500]
    lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in sample
    ]
    lengths = np.asarray(lengths)
    p95 = float(np.percentile(lengths, 95))
    chosen = int(np.ceil(p95 / 16.0) * 16)
    chosen = int(max(32, min(cap, chosen)))
    truncated = float((lengths > chosen).mean())
    return chosen, p95, truncated


def train_safe(model_key, train_df, val_df, test_df, max_len, hw, **kwargs):
    """Retry without checkpointing first, then smaller batches, if the GPU runs out."""
    attempts = [
        dict(batch_size=hw["batch_size"], grad_accum=hw["grad_accum"], optim_name=hw["optim_name"], max_len=max_len, gradient_checkpointing=False),
        dict(batch_size=hw["batch_size"], grad_accum=hw["grad_accum"], optim_name=hw["optim_name"], max_len=max_len, gradient_checkpointing=True),
        dict(batch_size=1, grad_accum=16, optim_name="adafactor", max_len=max_len, gradient_checkpointing=True),
        dict(batch_size=1, grad_accum=16, optim_name="adafactor", max_len=min(64, max_len), gradient_checkpointing=True),
    ]
    if kwargs.get("freeze_encoder"):
        for attempt in attempts:
            attempt["gradient_checkpointing"] = False
    last_error = None
    for attempt in attempts:
        try:
            return train_transformer_once(
                model_key,
                train_df,
                val_df,
                test_df,
                fp16=hw["fp16"],
                output_dir=os.path.join(RESULTS, "checkpoints", "runs"),
                **attempt,
                **kwargs,
            )
        except RuntimeError as err:
            last_error = err
            text = str(err).lower()
            oom = "out of memory" in text or ("alloc" in text and "cuda" in text)
            if not oom:
                raise
            print(f"[OOM] {model_key} failed at batch={attempt['batch_size']} max_len={attempt['max_len']} checkpointing={attempt['gradient_checkpointing']}. Retrying smaller.", flush=True)
            release_memory()
    raise last_error


def _guard(label, fn):
    """Record a failed run and let the rest of the experiment continue."""
    try:
        return fn()
    except Exception as err:
        import traceback
        text = traceback.format_exc()
        print(f"[Fail] {label}\n{text}", flush=True)
        _append_row(os.path.join(RESULTS, "failures.csv"), {
            "item": label,
            "error": f"{type(err).__name__}: {err}",
        })
        release_memory()
        return None


def _seed_row(model, seed, result, notes=""):
    metrics = result["metrics"]
    row = {
        "model": model,
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "val_macro_f1": result["val_macro_f1"],
        "train_seconds": result["train_seconds"],
        "params": result["params"],
        "notes": notes,
    }
    for name, scores in metrics["per_class"].items():
        row[f"f1_{name}"] = scores["f1"]
        row[f"precision_{name}"] = scores["precision"]
        row[f"recall_{name}"] = scores["recall"]
    return row


def _keep_predictions(model, frame, preds, probs, history, val_f1):
    pred_dir = os.path.join(RESULTS, "preds")
    os.makedirs(pred_dir, exist_ok=True)
    marker = os.path.join(pred_dir, f"{model}.val_f1")
    previous = -1.0
    if os.path.exists(marker):
        previous = float(open(marker, encoding="utf-8").read().strip() or -1)
    if val_f1 < previous:
        return
    out = frame[["text", "clean_text", "label", "label_text"]].copy()
    out["pred"] = preds
    out["pred_label"] = pd.Series(preds).map({0: "negative", 1: "neutral", 2: "positive"}).values
    if probs is not None:
        out["p_negative"] = probs[:, 0]
        out["p_neutral"] = probs[:, 1]
        out["p_positive"] = probs[:, 2]
    out.to_csv(os.path.join(pred_dir, f"{model}.csv"), index=False)
    hist_dir = os.path.join(RESULTS, "history")
    os.makedirs(hist_dir, exist_ok=True)
    with open(os.path.join(hist_dir, f"{model}.json"), "w", encoding="utf-8") as handle:
        json.dump(_jsonable(history), handle)
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(str(val_f1))
    from src.evaluate import plot_and_save_confusion_matrix
    from src.evaluate import evaluate_predictions

    metrics = evaluate_predictions(frame["label"].tolist(), preds)
    plot_and_save_confusion_matrix(
        metrics["confusion_matrix"],
        os.path.join(RESULTS, "plots", f"cm_{model}.png"),
        title=f"Confusion matrix: {model}",
    )
    if history:
        plot_training_curves(
            history,
            os.path.join(RESULTS, "plots", f"curves_{model}.png"),
            title=model,
        )


def _discard_run(result):
    trainer = result.get("trainer")
    release_memory(trainer, result.get("model"))
    run_root = os.path.join(RESULTS, "checkpoints", "runs")
    if os.path.isdir(run_root):
        shutil.rmtree(run_root, ignore_errors=True)


def write_summary():
    seeds = _read_csv(os.path.join(RESULTS, "metrics_seeds.csv"))
    if seeds.empty:
        return seeds
    rows = []
    for model, group in seeds.groupby("model"):
        rows.append({
            "model": model,
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(ddof=1) if len(group) > 1 else None,
            "macro_f1_mean": group["macro_f1"].mean(),
            "macro_f1_std": group["macro_f1"].std(ddof=1) if len(group) > 1 else None,
            "weighted_f1_mean": group["weighted_f1"].mean(),
            "weighted_f1_std": group["weighted_f1"].std(ddof=1) if len(group) > 1 else None,
            "train_seconds_mean": group["train_seconds"].mean(),
            "params": group["params"].iloc[-1],
            "n_seeds": len(group),
        })
    summary = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)
    plot_model_comparison(summary, os.path.join(RESULTS, "plots", "model_comparison.png"))
    return summary


def run_linear_models(train_df, val_df, test_df):
    path = os.path.join(RESULTS, "metrics_seeds.csv")
    done = _done_pairs(path)
    for name, kind in (("tfidf_logreg", "logreg"), ("tfidf_svm", "svm")):
        for seed in SEEDS:
            if (name, seed) in done:
                print(f"[Skip] {name} seed {seed}")
                continue
            result = _guard(f"{name} seed {seed}", lambda kind=kind, seed=seed: train_baseline(
                train_df, val_df, test_df,
                classifier_type=kind,
                seed=seed,
                output_dir=RESULTS,
                make_plots=(seed == SEEDS[0]),
            ))
            if result is None:
                continue
            _append_row(path, _seed_row(name, seed, result, notes="word 1-2 + char 2-5 TF-IDF"))
            if seed == SEEDS[0]:
                _keep_predictions(name, test_df, result["preds"], result["probs"], [], result["val_macro_f1"])
            release_memory(result.get("model"))


def run_bilstm_seeds(train_df, val_df, test_df):
    path = os.path.join(RESULTS, "metrics_seeds.csv")
    done = _done_pairs(path)
    for seed in SEEDS:
        if ("bilstm", seed) in done:
            print(f"[Skip] bilstm seed {seed}")
            continue
        result = _guard(f"bilstm seed {seed}", lambda seed=seed: train_bilstm(
            train_df, val_df, test_df,
            seed=seed, output_dir=RESULTS, make_plots=(seed == SEEDS[0]),
        ))
        if result is None:
            continue
        _append_row(path, _seed_row("bilstm", seed, result, notes="word BiLSTM, hidden 128"))
        _keep_predictions("bilstm", test_df, result["preds"], result["probs"], result["history"], result["val_macro_f1"])
        release_memory(result.get("model"))


def _lr_tag(lr) -> str:
    return f"{float(lr):.0e}"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def predict_classification(model, tokenizer, texts, max_len, batch_size=4):
    device = next(model.parameters()).device
    model.eval()
    pred_parts = []
    prob_parts = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        inputs = tokenizer(
            chunk, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        prob_parts.append(prob)
        pred_parts.extend(prob.argmax(axis=-1).tolist())
    if not prob_parts:
        return np.asarray([], dtype=int), np.zeros((0, 3))
    return np.asarray(pred_parts), np.concatenate(prob_parts, axis=0)


def _promote_lr_checkpoint(test_df, max_len, learning_rate):
    """Score the chosen learning-rate run on test once. Do not train it again."""
    seeds_path = os.path.join(RESULTS, "metrics_seeds.csv")
    if ("hingbert", 42) in _done_pairs(seeds_path):
        return
    tag = _lr_tag(learning_rate)
    ckpt = os.path.join(RESULTS, "checkpoints", "lr_grid", tag)
    if not os.path.isdir(ckpt):
        print("[LR grid] Winning checkpoint is not on disk. HingBERT seed 42 will train in the main loop.")
        return
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.evaluate import evaluate_predictions

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    preds, probs = predict_classification(
        model, tokenizer, test_df["clean_text"].tolist(), max_len, batch_size=4
    )
    metrics = evaluate_predictions(test_df["label"].tolist(), preds)
    grid = _read_csv(os.path.join(RESULTS, "lr_grid.csv"))
    match = grid.iloc[(grid["lr"].astype(float) - float(learning_rate)).abs().argmin()]
    hist_path = os.path.join(RESULTS, "history", f"lr_grid_{tag}.json")
    history = []
    if os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as handle:
            history = json.load(handle)
    optim_name = "" if "optim" not in match.index else str(match["optim"])
    result = {
        "metrics": metrics,
        "train_seconds": float(match["train_seconds"]),
        "params": parameter_count(model),
        "val_macro_f1": float(match["val_macro_f1"]),
    }
    notes = (
        f"lr={learning_rate}; max_len={max_len}; optim={optim_name}; "
        f"{MODEL_REGISTRY['hingbert']}; seed 42 is the selected learning-rate run, scored on test after selection"
    )
    _append_row(seeds_path, _seed_row("hingbert", 42, result, notes=notes))
    _keep_predictions("hingbert", test_df, preds, probs, history, result["val_macro_f1"])
    dest = os.path.join(RESULTS, "checkpoints", "hingbert")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(ckpt, dest)
    print(
        f"[LR grid] Recorded hingbert seed 42 from lr={learning_rate} "
        f"test macro-F1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    release_memory(model)
    grid_root = os.path.join(RESULTS, "checkpoints", "lr_grid")
    if os.path.isdir(grid_root):
        shutil.rmtree(grid_root, ignore_errors=True)


def select_learning_rate(train_df, val_df, test_df, max_len, hw):
    path = os.path.join(RESULTS, "lr_grid.csv")
    existing = _read_csv(path)
    done = set(existing["lr"].astype(float)) if not existing.empty and "lr" in existing.columns else set()
    for lr in LEARNING_RATES:
        if any(abs(lr - seen) < 1e-12 for seen in done):
            print(f"[Skip] lr grid {lr}")
            continue
        print(f"[LR grid] hingbert lr={lr}", flush=True)
        result = _guard(
            f"lr grid {lr}",
            lambda lr=lr: train_safe(
                "hingbert", train_df, val_df, test_df, max_len, hw,
                seed=42, epochs=3, learning_rate=lr, eval_test=False,
            ),
        )
        if result is None:
            continue
        tag = _lr_tag(lr)
        grid_dir = os.path.join(RESULTS, "checkpoints", "lr_grid", tag)
        save_checkpoint(result["trainer"], result["tokenizer"], grid_dir)
        hist_path = os.path.join(RESULTS, "history", f"lr_grid_{tag}.json")
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        with open(hist_path, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(result["history"]), handle)
        _append_row(path, {
            "model": "hingbert",
            "lr": lr,
            "val_macro_f1": result["val_macro_f1"],
            "val_accuracy": None,
            "train_seconds": result["train_seconds"],
            "optim": result["optim"],
        })
        done.add(lr)
        _discard_run(result)
    grid = _read_csv(path)
    if grid.empty:
        print("[LR grid] Every learning rate failed. Using 2e-5.")
        return 2e-5
    best = grid.sort_values("val_macro_f1", ascending=False).iloc[0]
    learning_rate = float(best["lr"])
    print(f"[LR grid] selected lr={learning_rate} val macro-F1={float(best['val_macro_f1']):.4f}", flush=True)
    _promote_lr_checkpoint(test_df, max_len, learning_rate)
    return learning_rate


def run_one_transformer(model_key, train_df, val_df, test_df, max_len, hw, learning_rate):
    """Train one model. A seed already listed in metrics_seeds.csv is skipped."""
    path = os.path.join(RESULTS, "metrics_seeds.csv")
    # Three seeds for HingBERT. The other models are one seed: a 4GB laptop
    # cannot repeat every 250k-vocab model three times.
    seeds = SEEDS if model_key == "hingbert" else [SEEDS[0]]
    for seed in seeds:
        done = _done_pairs(path)
        if (model_key, seed) in done:
            print(f"[Skip] {model_key} seed {seed}")
            continue
        print(f"[Train] {model_key} seed {seed} lr={learning_rate} max_len={max_len}", flush=True)
        result = _guard(
            f"{model_key} seed {seed}",
            lambda model_key=model_key, seed=seed: train_safe(
                model_key, train_df, val_df, test_df, max_len, hw,
                seed=seed, epochs=3, learning_rate=learning_rate, eval_test=True,
                use_class_weights=True, freeze_encoder=False,
            ),
        )
        if result is None:
            continue
        notes = f"lr={learning_rate}; max_len={max_len}; optim={result['optim']}; {MODEL_REGISTRY[model_key]}"
        _append_row(path, _seed_row(model_key, seed, result, notes=notes))
        _keep_predictions(
            model_key, test_df, result["preds"], result["probs"],
            result["history"], result["val_macro_f1"],
        )
        ckpt = os.path.join(RESULTS, "checkpoints", model_key)
        # Keep the validation-best seed. A later weaker seed must not overwrite it.
        marker = os.path.join(RESULTS, "preds", f"{model_key}.val_f1")
        current = float(open(marker, encoding="utf-8").read().strip())
        if abs(current - result["val_macro_f1"]) < 1e-8:
            save_checkpoint(result["trainer"], result["tokenizer"], ckpt)
        _discard_run(result)
        write_summary()


def run_transformers(train_df, val_df, test_df, max_len, hw, learning_rate):
    for model_key in TRANSFORMERS:
        run_one_transformer(model_key, train_df, val_df, test_df, max_len, hw, learning_rate)


def run_ablations(train_df, val_df, test_df, max_len, hw, learning_rate):
    path = os.path.join(RESULTS, "ablations.csv")
    existing = _read_csv(path)
    done = set(existing["ablation"]) if not existing.empty and "ablation" in existing.columns else set()
    seeds = _read_csv(os.path.join(RESULTS, "metrics_seeds.csv"))
    full = seeds[(seeds["model"] == "hingbert") & (seeds["seed"] == 42)]
    if "full" not in done and not full.empty:
        row = full.iloc[0]
        _append_row(path, {
            "ablation": "full",
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "val_macro_f1": row["val_macro_f1"],
            "train_seconds": row["train_seconds"],
            "notes": "preprocessed, emojis kept, class-weighted, full fine-tune, seed 42",
        })
        done.add("full")

    variants = [
        ("raw_text", dict(raw=True, strip_emojis=False, use_class_weights=True, freeze_encoder=False)),
        ("no_emoji", dict(raw=False, strip_emojis=True, use_class_weights=True, freeze_encoder=False)),
        ("no_class_weights", dict(raw=False, strip_emojis=False, use_class_weights=False, freeze_encoder=False)),
        ("frozen_encoder", dict(raw=False, strip_emojis=False, use_class_weights=True, freeze_encoder=True)),
    ]
    for name, spec in variants:
        if name in done:
            print(f"[Skip] ablation {name}")
            continue
        print(f"[Ablation] {name}")
        tr = attach_clean(train_df, strip_emojis=spec["strip_emojis"], raw=spec["raw"])
        va = attach_clean(val_df, strip_emojis=spec["strip_emojis"], raw=spec["raw"])
        te = attach_clean(test_df, strip_emojis=spec["strip_emojis"], raw=spec["raw"])
        result = _guard(
            f"ablation {name}",
            lambda spec=spec, tr=tr, va=va, te=te: train_safe(
                "hingbert", tr, va, te, max_len, hw,
                seed=42, epochs=3, learning_rate=learning_rate, eval_test=True,
                use_class_weights=spec["use_class_weights"],
                freeze_encoder=spec["freeze_encoder"],
            ),
        )
        if result is None:
            continue
        _append_row(path, {
            "ablation": name,
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "val_macro_f1": result["val_macro_f1"],
            "train_seconds": result["train_seconds"],
            "notes": "hingbert seed 42",
        })
        _discard_run(result)


def run_transfer(test_like_df):
    """Score the best SentiMix checkpoint on the second dataset. No extra training."""
    summary = _read_csv(os.path.join(RESULTS, "metrics.csv"))
    transformers = summary[summary["model"].isin(TRANSFORMERS)]
    if transformers.empty:
        print("[Transfer] No transformer summary yet.")
        return
    winner = transformers.sort_values("macro_f1_mean", ascending=False).iloc[0]["model"]
    ckpt = os.path.join(RESULTS, "checkpoints", winner)
    if not os.path.isdir(ckpt):
        print(f"[Transfer] Missing checkpoint for {winner}")
        return
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    texts = test_like_df["clean_text"].tolist()
    preds = []
    batch = 4
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        inputs = tokenizer(chunk, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    from src.evaluate import evaluate_predictions

    metrics = evaluate_predictions(test_like_df["label"].tolist(), preds)
    row = {
        "model": winner,
        "n": len(test_like_df),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "source": "Mendeley Hinglish.csv DOI 10.17632/wtp7htm3ss.1",
    }
    for name, scores in metrics["per_class"].items():
        row[f"f1_{name}"] = scores["f1"]
    pd.DataFrame([row]).to_csv(os.path.join(RESULTS, "transfer.csv"), index=False)
    print(f"[Transfer] {winner} macro-F1={metrics['macro_f1']:.4f} on {len(test_like_df)} rows")
    best = os.path.join(RESULTS, "checkpoints", "best")
    if os.path.isdir(best):
        shutil.rmtree(best)
    shutil.copytree(ckpt, best)
    with open(os.path.join(RESULTS, "best_model.json"), "w", encoding="utf-8") as handle:
        json.dump({"model": winner, "path": "results/checkpoints/best", "hf_name": MODEL_REGISTRY[winner]}, handle, indent=2)
    release_memory(model)


def _checkpoint_preds(path, texts, max_len):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    if torch.cuda.is_available():
        model.to("cuda")
    preds, probs = predict_classification(model, tokenizer, list(texts), max_len, batch_size=8)
    model.to("cpu")
    del model
    release_memory()
    return preds, probs


def _flip_slice_scores(frame, preds, weights):
    from sklearn.metrics import f1_score

    from src.flip import analyze

    gold = frame["label"].to_numpy()
    negation = np.asarray([
        analyze(text, weights)["has_negation"] == 1.0 for text in frame["clean_text"].astype(str)
    ])
    return {
        "macro_f1": float(f1_score(gold, preds, average="macro", zero_division=0)),
        "negation_macro_f1": float(f1_score(gold[negation], preds[negation], average="macro", zero_division=0)),
        "negation_n": int(negation.sum()),
    }


def run_flip_hingbert(train_df, val_df, test_df, max_len, hw, learning_rate):
    """One more HingBERT epoch on the official train set plus mechanical flip rows.

    The checkpoint is kept only when validation overall macro-F1 stays within
    0.005 of the current HingBERT and the validation negation slice gains at
    least 0.01. The test split is scored only after that decision.
    """
    out_csv = os.path.join(RESULTS, "flip_model_metrics.csv")
    if os.path.exists(out_csv):
        print("[Skip] hingbert flip continuation")
        return
    ckpt = os.path.join(RESULTS, "checkpoints", "hingbert")
    if not os.path.isdir(ckpt):
        print("[Flip model] HingBERT checkpoint is missing. Continuation skipped.")
        return
    from src.flip import scope_weights
    from src.flip_add import build_extra
    from src.system import load_system

    system = load_system()
    if system is None:
        print("[Flip model] Character system is missing. Continuation skipped.")
        return
    weights = scope_weights(system.weights)
    extra = build_extra(train_df, weights)
    added = pd.DataFrame({
        "text": extra["clean_text"].values,
        "clean_text": extra["clean_text"].values,
        "label": extra["label"].astype(int).values,
        "label_text": extra["label_text"].values,
    })
    mixed = pd.concat([train_df, added], ignore_index=True)
    print(f"[Train] hingbert flip continuation extra={len(added)} lr=1e-05 epochs=1", flush=True)
    old_val, _ = _checkpoint_preds(ckpt, val_df["clean_text"], max_len)
    before = _flip_slice_scores(val_df, old_val, weights)
    result = train_safe(
        "hingbert",
        mixed,
        val_df,
        test_df,
        max_len,
        hw,
        seed=42,
        epochs=1,
        learning_rate=1e-5,
        eval_test=False,
        init_from=ckpt,
    )
    new_val, _ = predict_classification(
        result["trainer"].model,
        result["tokenizer"],
        val_df["clean_text"].tolist(),
        max_len,
        batch_size=8,
    )
    after = _flip_slice_scores(val_df, new_val, weights)
    gain = after["negation_macro_f1"] - before["negation_macro_f1"]
    kept = after["macro_f1"] + 0.005 >= before["macro_f1"] and gain >= 0.01
    row = {
        "model": "hingbert_flip",
        "kept": bool(kept),
        "extra_rows": int(len(added)),
        "val_macro_f1_before": before["macro_f1"],
        "val_macro_f1_after": after["macro_f1"],
        "val_negation_macro_f1_before": before["negation_macro_f1"],
        "val_negation_macro_f1_after": after["negation_macro_f1"],
        "negation_gain": gain,
        "base_learning_rate": learning_rate,
        "continuation_learning_rate": 1e-5,
        "epochs": 1,
    }
    if kept:
        test_preds, test_probs = predict_classification(
            result["trainer"].model,
            result["tokenizer"],
            test_df["clean_text"].tolist(),
            max_len,
            batch_size=8,
        )
        from src.evaluate import evaluate_predictions
        test_metrics = evaluate_predictions(test_df["label"].tolist(), test_preds)
        test_slice = _flip_slice_scores(test_df, test_preds, weights)
        row.update({
            "test_macro_f1": test_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_negation_macro_f1": test_slice["negation_macro_f1"],
        })
        save_checkpoint(result["trainer"], result["tokenizer"], os.path.join(RESULTS, "checkpoints", "hingbert_flip"))
        _keep_predictions("hingbert_flip", test_df, test_preds, test_probs, result["history"], after["macro_f1"])
        _append_row(os.path.join(RESULTS, "metrics_seeds.csv"), _seed_row(
            "hingbert_flip", 42,
            {"metrics": test_metrics, "train_seconds": result["train_seconds"], "params": result["params"], "val_macro_f1": after["macro_f1"]},
            notes="one epoch continued from hingbert on train plus mechanical flip rows; kept on validation",
        ))
        write_summary()
    else:
        row["test_macro_f1"] = None
        print("[Flip model] Validation did not keep the continuation. HingBERT checkpoint unchanged.")
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    _discard_run(result)
    print(f"[Flip model] kept={kept} negation_gain={gain:.4f}")


def write_error_report():
    summary = _read_csv(os.path.join(RESULTS, "metrics.csv"))
    transformers = summary[summary["model"].isin(TRANSFORMERS)]
    if transformers.empty:
        return
    winner = transformers.sort_values("macro_f1_mean", ascending=False).iloc[0]["model"]
    pred_path = os.path.join(RESULTS, "preds", f"{winner}.csv")
    if not os.path.exists(pred_path):
        return
    preds = pd.read_csv(pred_path)
    prob_cols = [c for c in ("p_negative", "p_neutral", "p_positive") if c in preds.columns]
    probs = preds[prob_cols].to_numpy() if prob_cols else None
    perform_error_analysis(
        texts=preds["text"].tolist(),
        y_true=preds["label"].tolist(),
        y_pred=preds["pred"].tolist(),
        y_probs=probs,
        output_csv=os.path.join(RESULTS, "errors_best.csv"),
        max_samples=50,
    )


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.chdir(ROOT)
    if not os.path.exists(os.path.join(RESULTS, "metrics_seeds.csv")):
        old_metrics = os.path.join(RESULTS, "metrics.csv")
        if os.path.exists(old_metrics):
            os.remove(old_metrics)
    print("[Data] Loading official SentiMix Hinglish split")
    train_df, val_df, test_df = load_official_sentimix(os.path.join(ROOT, "data", "raw"))
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        keep = [c for c in ("uid", "text", "sentiment", "label", "label_text", "cmi", "n_tokens", "n_hin", "n_eng") if c in frame.columns]
        frame[keep].to_csv(os.path.join(ROOT, "data", f"sentimix_{name}.csv"), index=False)

    run_eda({"train": train_df, "validation": val_df, "test": test_df}, RESULTS)
    train_df = attach_clean(train_df)
    val_df = attach_clean(val_df)
    test_df = attach_clean(test_df)

    hw = hardware_config()
    max_len, p95, truncated = choose_max_len(
        pd.concat([train_df["clean_text"], val_df["clean_text"], test_df["clean_text"]]),
        hw["max_len_cap"],
    )
    config = {
        "seeds": SEEDS,
        "seeds_note": "HingBERT is trained with all three seeds. mBERT, MuRIL, IndicBERTv2, and XLM-RoBERTa use seed 42 only.",
        "max_len": max_len,
        "subword_p95_mbert": p95,
        "fraction_over_max_len_in_sample": truncated,
        "hardware": hw,
        "official_split": "SemEval-2020 Task 9 Hinglish, Zenodo 3974927",
        "test_used_for": "final evaluation only; learning rate chosen on validation",
    }
    with open(os.path.join(RESULTS, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(f"[Setup] max_len={max_len} (mBERT subword p95={p95:.1f}, sample truncated={truncated:.3f})")
    print(f"[Setup] hardware={hw}")

    run_linear_models(train_df, val_df, test_df)
    run_bilstm_seeds(train_df, val_df, test_df)
    write_summary()

    learning_rate = select_learning_rate(train_df, val_df, test_df, max_len, hw)
    config["learning_rate"] = learning_rate
    with open(os.path.join(RESULTS, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    run_transformers(train_df, val_df, test_df, max_len, hw, learning_rate)
    run_ablations(train_df, val_df, test_df, max_len, hw, learning_rate)
    write_summary()
    write_error_report()

    if os.path.exists(TRANSFER_CSV):
        blocked = pd.concat([train_df["text"], val_df["text"], test_df["text"]]).tolist()
        transfer = load_transfer_csv(TRANSFER_CSV, blocked_texts=blocked)
        transfer = attach_clean(transfer)
        transfer.to_csv(os.path.join(ROOT, "data", "transfer_hinglish.csv"), index=False)
        run_transfer(transfer)
    _guard(
        "hingbert flip",
        lambda: run_flip_hingbert(train_df, val_df, test_df, max_len, hw, learning_rate),
    )
    print("[Done] Results are in", RESULTS)


if __name__ == "__main__":
    main()
