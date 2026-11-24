"""Training loops for the TF-IDF baselines, a BiLSTM, and the transformer runs.

The official SentiMix split is fixed. Seeds change initialization, not the split.
"""

import argparse
import gc
import inspect
import os
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

try:
    from src.data import (
        HinglishSentimentDataset,
        compute_class_weights_tensor,
        load_dataset_from_file,
        split_data,
    )
    from src.evaluate import (
        evaluate_predictions,
        compute_metrics_for_trainer,
        plot_and_save_confusion_matrix,
        perform_error_analysis,
    )
    from src.models import (
        BiLSTMClassifier,
        MODEL_REGISTRY,
        WeightedTrainer,
        build_tfidf_baseline,
        load_transformer_and_tokenizer,
    )
    from src.preprocess import preprocess_corpus
except ImportError:
    from data import (
        HinglishSentimentDataset,
        compute_class_weights_tensor,
        load_dataset_from_file,
        split_data,
    )
    from evaluate import (
        evaluate_predictions,
        compute_metrics_for_trainer,
        plot_and_save_confusion_matrix,
        perform_error_analysis,
    )
    from models import (
        BiLSTMClassifier,
        MODEL_REGISTRY,
        WeightedTrainer,
        build_tfidf_baseline,
        load_transformer_and_tokenizer,
    )
    from preprocess import preprocess_corpus


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def release_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _scores_from_preds(frame, preds, probs, output_dir, tag):
    metrics = evaluate_predictions(frame["label"].tolist(), preds)
    cm_path = os.path.join(output_dir, "plots", f"cm_{tag}.png")
    plot_and_save_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=cm_path,
        title=f"Confusion matrix: {tag}",
    )
    err_path = os.path.join(output_dir, f"errors_{tag}.csv")
    perform_error_analysis(
        texts=frame["text"].tolist(),
        y_true=frame["label"].tolist(),
        y_pred=list(preds),
        y_probs=probs,
        output_csv=err_path,
    )
    return metrics


def train_baseline(
    train_df,
    val_df,
    test_df,
    classifier_type: str = "logreg",
    use_class_weights: bool = True,
    seed: int = 42,
    output_dir: str = "results",
    make_plots: bool = True,
):
    """Fit TF-IDF + logistic regression or linear SVM on a fixed split.

    Validation is scored and stored. It is not used to pick features or C.
    """
    seed_everything(seed)
    started = time.perf_counter()
    model = build_tfidf_baseline(classifier_type=classifier_type, use_class_weights=use_class_weights)
    # LogisticRegression / LinearSVC accept random_state at construction.
    clf = model.named_steps["classifier"]
    if hasattr(clf, "set_params"):
        clf.set_params(random_state=seed)
    model.fit(train_df["clean_text"], train_df["label"])
    elapsed = time.perf_counter() - started

    val_preds = model.predict(val_df["clean_text"])
    val_f1 = evaluate_predictions(val_df["label"].tolist(), val_preds)["macro_f1"]
    preds = model.predict(test_df["clean_text"])
    probs = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probs = model.predict_proba(test_df["clean_text"])
    metrics = evaluate_predictions(test_df["label"].tolist(), preds)
    try:
        n_features = int(len(model.named_steps["features"].get_feature_names_out()))
    except Exception:
        n_features = -1
    if make_plots:
        _scores_from_preds(test_df, preds, probs, output_dir, f"baseline_{classifier_type}")
    print(
        f"[Baseline {classifier_type} seed {seed}] "
        f"acc={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f} "
        f"features={n_features} seconds={elapsed:.1f}"
    )
    return {
        "model": model,
        "metrics": metrics,
        "preds": preds,
        "probs": probs,
        "train_seconds": elapsed,
        "params": n_features,
        "val_macro_f1": val_f1,
    }


def _encode_words(texts, vocab, max_len):
    unk = vocab["<unk>"]
    rows = []
    for text in texts:
        ids = [vocab.get(tok, unk) for tok in str(text).split()][:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def build_word_vocab(texts, min_freq: int = 2, max_size: int = 30000):
    counts = Counter()
    for text in texts:
        counts.update(str(text).split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in counts.most_common(max_size):
        if count < min_freq:
            break
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def train_bilstm(
    train_df,
    val_df,
    test_df,
    seed: int = 42,
    epochs: int = 8,
    batch_size: int = 64,
    max_len: int = 48,
    lr: float = 1e-3,
    use_class_weights: bool = True,
    output_dir: str = "results",
    make_plots: bool = True,
):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = build_word_vocab(train_df["clean_text"])
    model = BiLSTMClassifier(vocab_size=len(vocab)).to(device)
    weight = None
    if use_class_weights:
        weight = compute_class_weights_tensor(train_df["label"].tolist(), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def loader(frame, shuffle):
        x = _encode_words(frame["clean_text"], vocab, max_len)
        y = torch.tensor(frame["label"].tolist(), dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(train_df, True)
    val_x = _encode_words(val_df["clean_text"], vocab, max_len).to(device)
    val_y = val_df["label"].tolist()
    best_state = None
    best_val = -1.0
    stale = 0
    started = time.perf_counter()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_y)
            seen += len(batch_y)
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
        val_metrics = evaluate_predictions(val_y, val_preds)
        history.append({
            "epoch": epoch,
            "loss": total / max(seen, 1),
            "eval_macro_f1": val_metrics["macro_f1"],
        })
        print(f"[BiLSTM seed {seed}] epoch {epoch} val macro-F1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    elapsed = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_x = _encode_words(test_df["clean_text"], vocab, max_len).to(device)
    with torch.no_grad():
        test_logits = model(test_x)
        probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
    metrics = evaluate_predictions(test_df["label"].tolist(), preds)
    if make_plots:
        _scores_from_preds(test_df, preds, probs, output_dir, "bilstm")
    print(f"[BiLSTM seed {seed}] test macro-F1={metrics['macro_f1']:.4f}")
    return {
        "metrics": metrics,
        "preds": preds,
        "probs": probs,
        "train_seconds": elapsed,
        "params": parameter_count(model),
        "val_macro_f1": best_val,
        "history": history,
        "vocab": vocab,
        "model": model,
    }


def _training_arguments(
    output_dir, learning_rate, batch_size, epochs, fp16, grad_accum, optim_name,
    gradient_checkpointing=True,
):
    kwargs = dict(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=0.1,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=fp16,
        logging_strategy="steps",
        logging_steps=25,
        save_total_limit=1,
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=0,
        optim=optim_name,
        gradient_checkpointing=gradient_checkpointing,
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**kwargs)


def _padding_collator(tokenizer):
    from transformers import DataCollatorWithPadding

    params = inspect.signature(DataCollatorWithPadding.__init__).parameters
    if "tokenizer" in params:
        return DataCollatorWithPadding(tokenizer=tokenizer)
    return DataCollatorWithPadding(processing_class=tokenizer)


def _make_trainer(model, args, train_ds, val_ds, tokenizer, class_weights):
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=_padding_collator(tokenizer),
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
    # WeightedTrainer.__init__ only exposes *args/**kwargs, so inspect Trainer.
    trainer_params = inspect.signature(WeightedTrainer.__bases__[0].__init__).parameters
    if "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return WeightedTrainer(**kwargs)


def train_transformer_once(
    model_name_or_key: str,
    train_df,
    val_df,
    test_df,
    seed: int = 42,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_len: int = 96,
    fp16: bool = True,
    use_class_weights: bool = True,
    freeze_encoder: bool = False,
    optim_name: str = "adamw_torch",
    output_dir: str = "results/checkpoints",
    eval_test: bool = True,
    gradient_checkpointing: bool = True,
    init_from: str = None,
):
    """Fine-tune one transformer on a fixed split and one seed."""
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(fp16 and torch.cuda.is_available())
    if optim_name == "adamw_bnb_8bit":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("[Train] bitsandbytes is not installed. Using Adafactor.")
            optim_name = "adafactor"

    if init_from:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(init_from)
        model = AutoModelForSequenceClassification.from_pretrained(init_from)
        print(f"[Train] Continued from {init_from}")
    else:
        tokenizer, model = load_transformer_and_tokenizer(
            model_name_or_key, num_labels=3, freeze_encoder=freeze_encoder
        )
    n_params = parameter_count(model)

    def tokenize(frame):
        return tokenizer(
            list(frame["clean_text"]),
            truncation=True,
            max_length=max_len,
        )

    train_ds = HinglishSentimentDataset(tokenize(train_df), list(train_df["label"]))
    val_ds = HinglishSentimentDataset(tokenize(val_df), list(val_df["label"]))
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights_tensor(train_df["label"].tolist(), device=device)

    run_dir = os.path.join(output_dir, f"{model_name_or_key}_seed{seed}")
    use_ckpt = bool(gradient_checkpointing and not freeze_encoder)
    try:
        args = _training_arguments(
            run_dir, learning_rate, batch_size, epochs, use_fp16, grad_accum, optim_name,
            gradient_checkpointing=use_ckpt,
        )
    except (TypeError, ValueError) as err:
        if optim_name == "adafactor" or "optim" not in str(err).lower():
            raise
        print(f"[Train] Optimizer {optim_name} was rejected ({err}). Using Adafactor.")
        optim_name = "adafactor"
        args = _training_arguments(
            run_dir, learning_rate, batch_size, epochs, use_fp16, grad_accum, optim_name,
            gradient_checkpointing=use_ckpt,
        )
    trainer = _make_trainer(model, args, train_ds, val_ds, tokenizer, class_weights)
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    val_metrics = trainer.evaluate()
    val_f1 = float(val_metrics.get("eval_macro_f1", 0.0))

    test_metrics = None
    preds = None
    probs = None
    if eval_test:
        test_ds = HinglishSentimentDataset(tokenize(test_df), list(test_df["label"]))
        pred_out = trainer.predict(test_ds)
        logits = pred_out.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = probs.argmax(axis=-1)
        test_metrics = evaluate_predictions(test_df["label"].tolist(), preds)
        print(
            f"[{model_name_or_key} seed {seed}] val macro-F1={val_f1:.4f} "
            f"test macro-F1={test_metrics['macro_f1']:.4f} seconds={elapsed:.1f}"
        )
    else:
        print(f"[{model_name_or_key} seed {seed}] val macro-F1={val_f1:.4f} seconds={elapsed:.1f}")

    return {
        "trainer": trainer,
        "tokenizer": tokenizer,
        "metrics": test_metrics,
        "preds": preds,
        "probs": probs,
        "train_seconds": elapsed,
        "params": n_params,
        "val_macro_f1": val_f1,
        "history": list(trainer.state.log_history),
        "optim": optim_name,
    }


def save_checkpoint(trainer, tokenizer, path: str):
    os.makedirs(path, exist_ok=True)
    trainer.save_model(path)
    tokenizer.save_pretrained(path)
    print(f"[Train] Saved checkpoint to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train one Hinglish sentiment model")
    parser.add_argument("--data_path", type=str, default="data/sample_hinglish.csv")
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_logreg",
        choices=["baseline_logreg", "baseline_svm", "bilstm", *MODEL_REGISTRY.keys()],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_dataset_from_file(args.data_path, dedup=True)
    df["clean_text"] = preprocess_corpus(df["text"])
    train_df, val_df, test_df = split_data(df, test_size=0.15, val_size=0.15, random_state=42)
    if args.model.startswith("baseline"):
        kind = "logreg" if "logreg" in args.model else "svm"
        train_baseline(train_df, val_df, test_df, kind, seed=args.seed, output_dir=args.output_dir)
    elif args.model == "bilstm":
        train_bilstm(train_df, val_df, test_df, seed=args.seed, epochs=args.epochs, output_dir=args.output_dir)
    else:
        train_transformer_once(
            args.model, train_df, val_df, test_df,
            seed=args.seed, epochs=args.epochs, learning_rate=args.lr,
            batch_size=args.batch_size, max_len=args.max_len,
            output_dir=os.path.join(args.output_dir, "checkpoints"),
        )


if __name__ == "__main__":
    main()
