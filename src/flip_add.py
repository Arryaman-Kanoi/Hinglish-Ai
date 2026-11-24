"""Add mechanical flips to the character model.

Training rows are built only from the official training split. A clear positive
tweet gains one copy with nahi inserted before its positive word, labeled
negative. A clear negative tweet whose negation reverses that positive word
gains one copy with the negation removed, labeled positive. Real training
tweets where the scope already matches the gold label are repeated once.

The character model is refit on those rows. The stacker is fit on 70% of
validation. The other 30% decides whether the new model replaces the saved
one. The test split is scored only after that decision. HingBERT is not
loaded, so this can run while another fine-tune holds the GPU.
"""

import json
import os
import re
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

try:
    from src.flip import (
        CLOSED,
        CONTRAST,
        NEGATION,
        RESULTS,
        TAG_TOKENS,
        _strip_tag,
        analyze,
        scope_weights,
    )
    from src.lexicon import tokens_of
    from src.models import build_tfidf_baseline
    from src.system import (
        DATA,
        MARGINS,
        SYSTEM,
        _align_proba,
        _apply_margin,
        _clean_frame,
        _lexicon_matrix,
        _scores,
        _stack_matrix,
        load_system,
    )
except ImportError:
    from flip import (
        CLOSED,
        CONTRAST,
        NEGATION,
        RESULTS,
        TAG_TOKENS,
        _strip_tag,
        analyze,
        scope_weights,
    )
    from lexicon import tokens_of
    from models import build_tfidf_baseline
    from system import (
        DATA,
        MARGINS,
        SYSTEM,
        _align_proba,
        _apply_margin,
        _clean_frame,
        _lexicon_matrix,
        _scores,
        _stack_matrix,
        load_system,
    )

REMOVABLE = NEGATION - TAG_TOKENS - {"no", "na"}
HOLDOUT_FRACTION = 0.30
OVERALL_SLACK = 0.005
SLICE_GAIN = 0.01
MAX_INSERTS = 2000
ID2NAME = {0: "negative", 1: "neutral", 2: "positive"}


def _carrier(text, weights):
    """Return the single strong positive word, when the rest of the line is not polar."""
    body, _ = _strip_tag(tokens_of(text))
    if any(tok in NEGATION and tok not in TAG_TOKENS for tok in body):
        return None
    if any(tok in CONTRAST for tok in body):
        return None
    scored = []
    for tok in body:
        if tok in CLOSED or tok in NEGATION:
            continue
        weight = weights.get(tok, 0.0)
        if weight == 0.0:
            continue
        scored.append((weight, tok))
    if not scored:
        return None
    scored.sort(reverse=True)
    best, word = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best < 1.5 or second >= 0.5:
        return None
    if any(weight <= -1.0 for weight, _ in scored):
        return None
    return word


def _insert_nahi(text, word):
    match = re.search(rf"\b{re.escape(word)}\b", str(text), flags=re.IGNORECASE)
    if match is None:
        return None
    return str(text)[:match.start()] + "nahi " + str(text)[match.start():]


def _remove_one_negation(text):
    pattern = re.compile(r"\b(" + "|".join(sorted(REMOVABLE, key=len, reverse=True)) + r")\b", re.IGNORECASE)
    hits = list(pattern.finditer(str(text)))
    if len(hits) != 1:
        return None
    hit = hits[0]
    edited = (str(text)[:hit.start()] + " " + str(text)[hit.end():]).strip()
    edited = re.sub(r"\s+", " ", edited)
    if pattern.search(edited):
        return None
    return edited


def build_extra(train_df, weights):
    existing = set(train_df["clean_text"].astype(str))
    rows = []
    for text, label in zip(train_df["clean_text"].astype(str), train_df["label"].astype(int)):
        scope = analyze(text, weights)
        gold_sign = -1 if label == 0 else (1 if label == 2 else 0)
        if gold_sign and scope["scoped_sign"] == gold_sign and scope["kind"] in {"negation", "contrast", "echo"}:
            rows.append({"clean_text": text, "label": label, "edit": "repeat_real_flip"})
        if label == 2:
            word = _carrier(text, weights)
            if word is None:
                continue
            edited = _insert_nahi(text, word)
            if edited and edited not in existing:
                rows.append({"clean_text": edited, "label": 0, "edit": f"insert_nahi:{word}"})
        elif label == 0 and scope["kind"] == "negation" and scope["surface_sign"] > 0 and scope["scoped_sign"] < 0:
            edited = _remove_one_negation(text)
            if edited and edited not in existing:
                rows.append({"clean_text": edited, "label": 2, "edit": "remove_negation"})
    extra = pd.DataFrame(rows)
    if extra.empty:
        return extra
    inserts = extra[extra["edit"].str.startswith("insert_nahi")]
    if len(inserts) > MAX_INSERTS:
        inserts = inserts.head(MAX_INSERTS)
        extra = pd.concat([
            extra[extra["edit"] == "repeat_real_flip"],
            extra[extra["edit"] == "remove_negation"],
            inserts,
        ], ignore_index=True)
    extra["label_text"] = extra["label"].map(ID2NAME)
    return extra


def _negation_mask(texts, weights):
    return np.asarray([analyze(text, weights)["has_negation"] == 1.0 for text in texts])


def _fit_router(tfidf_p, lex_p, texts, labels):
    router = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    features = _stack_matrix(tfidf_p, lex_p, texts)
    router.fit(features, labels)
    proba = _align_proba(router.predict_proba(features), router.classes_)
    best_t = 0.0
    best_f1 = -1.0
    for threshold in MARGINS:
        score = f1_score(labels, _apply_margin(proba, threshold), average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(threshold)
    return router, best_t


def _bundle_pred(tfidf, lex_clf, router, margin, weights, texts):
    lex_x, _ = _lexicon_matrix(texts, weights)
    lex_p = _align_proba(lex_clf.predict_proba(lex_x), lex_clf.classes_)
    tfidf_p = _align_proba(tfidf.predict_proba(list(texts)), tfidf.named_steps["classifier"].classes_)
    features = _stack_matrix(tfidf_p, lex_p, texts)
    proba = _align_proba(router.predict_proba(features), router.classes_)
    return _apply_margin(proba, margin)


def add_flips():
    system = load_system()
    if system is None:
        raise RuntimeError("results/system/router.joblib is missing.")
    train_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_train.csv")))
    val_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_val.csv")))
    weights = scope_weights(system.weights)
    extra = build_extra(train_df, weights)
    os.makedirs(os.path.join(RESULTS, "flip_added"), exist_ok=True)
    extra.to_csv(os.path.join(RESULTS, "flip_added", "train_extra.csv"), index=False)
    print("[Flip add] edits")
    print(extra["edit"].str.replace(r":.*", "", regex=True).value_counts().to_string())

    combined_text = pd.concat([train_df["clean_text"], extra["clean_text"]], ignore_index=True)
    combined_label = pd.concat([train_df["label"], extra["label"]], ignore_index=True)
    tfidf = build_tfidf_baseline("logreg", use_class_weights=True)
    tfidf.fit(combined_text, combined_label)

    val_texts = val_df["clean_text"].tolist()
    val_gold = val_df["label"].to_numpy()
    negation = _negation_mask(val_texts, weights)
    rng = np.random.RandomState(42)
    order = rng.permutation(len(val_df))
    hold_n = int(round(len(val_df) * HOLDOUT_FRACTION))
    hold = np.sort(order[:hold_n])
    fit = np.sort(order[hold_n:])

    def parts(model, texts):
        lex_x, _ = _lexicon_matrix(texts, system.weights)
        lex_p = _align_proba(system.lex_clf.predict_proba(lex_x), system.lex_clf.classes_)
        tfidf_p = _align_proba(model.predict_proba(list(texts)), model.named_steps["classifier"].classes_)
        return tfidf_p, lex_p

    fit_texts = [val_texts[i] for i in fit]
    fit_tfidf, fit_lex = parts(tfidf, fit_texts)
    router, margin = _fit_router(fit_tfidf, fit_lex, fit_texts, val_gold[fit])
    hold_texts = [val_texts[i] for i in hold]
    new_hold = _bundle_pred(tfidf, system.lex_clf, router, margin, system.weights, hold_texts)
    old_hold = _bundle_pred(
        system.tfidf, system.lex_clf, system.router, system.margin, system.weights, hold_texts
    )
    new_all = _scores("flip_added", val_gold[hold], new_hold)
    old_all = _scores("current", val_gold[hold], old_hold)
    new_neg = _scores("flip_added", val_gold[hold][negation[hold]], new_hold[negation[hold]])
    old_neg = _scores("current", val_gold[hold][negation[hold]], old_hold[negation[hold]])
    gain = new_neg["macro_f1"] - old_neg["macro_f1"]
    kept = new_all["macro_f1"] + OVERALL_SLACK >= old_all["macro_f1"] and gain >= SLICE_GAIN
    print(
        f"[Flip add] holdout current={old_all['macro_f1']:.4f} added={new_all['macro_f1']:.4f} "
        f"negation {old_neg['macro_f1']:.4f} -> {new_neg['macro_f1']:.4f} gain={gain:.4f} kept={kept}"
    )
    decision = {
        "kept": bool(kept),
        "holdout_current_macro_f1": old_all["macro_f1"],
        "holdout_added_macro_f1": new_all["macro_f1"],
        "holdout_current_negation_macro_f1": old_neg["macro_f1"],
        "holdout_added_negation_macro_f1": new_neg["macro_f1"],
        "negation_gain": gain,
        "extra_rows": int(len(extra)),
        "where": "character model training rows. HingBERT weights are unchanged.",
    }
    if not kept:
        with open(os.path.join(RESULTS, "flip_added", "decision.json"), "w", encoding="utf-8") as handle:
            json.dump(decision, handle, indent=2)
        print("[Flip add] not kept. Saved model is unchanged.")
        return decision

    # Refit the stacker on all validation rows. Test labels are not used.
    full_tfidf, full_lex = parts(tfidf, val_texts)
    router, margin = _fit_router(full_tfidf, full_lex, val_texts, val_gold)
    test_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_test.csv")))
    test_texts = test_df["clean_text"].tolist()
    test_gold = test_df["label"].to_numpy()
    test_neg = _negation_mask(test_texts, weights)
    new_test = _bundle_pred(tfidf, system.lex_clf, router, margin, system.weights, test_texts)
    old_test = _bundle_pred(
        system.tfidf, system.lex_clf, system.router, system.margin, system.weights, test_texts
    )
    summary = []
    for name, pred in (("current", old_test), ("flip_added", new_test)):
        for slice_name, mask in (("all", np.ones(len(test_gold), dtype=bool)), ("negation", test_neg)):
            scores = _scores(name, test_gold[mask], pred[mask])
            scores["slice"] = slice_name
            scores["n"] = int(mask.sum())
            summary.append(scores)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS, "flip_added", "test.csv"), index=False)

    backup = os.path.join(RESULTS, "system_before_flip")
    os.makedirs(backup, exist_ok=True)
    for name in ("tfidf.joblib", "router.joblib", "meta.json"):
        source = os.path.join(SYSTEM, name)
        if os.path.exists(source) and not os.path.exists(os.path.join(backup, name)):
            shutil.copy2(source, os.path.join(backup, name))
    import joblib
    joblib.dump(tfidf, os.path.join(SYSTEM, "tfidf.joblib"))
    joblib.dump(router, os.path.join(SYSTEM, "router.joblib"))
    meta_path = os.path.join(SYSTEM, "meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    meta["neutral_margin"] = margin
    meta["flip_rows_added"] = int(len(extra))
    meta["fit_on"] = meta.get("fit_on", "") + "; character model refit with mechanical flip rows"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    decision["test_scored"] = True
    decision["neutral_margin"] = margin
    with open(os.path.join(RESULTS, "flip_added", "decision.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    print("[Flip add] kept. Character model and stacker replaced.")
    print(pd.DataFrame(summary).to_string(index=False))
    return decision


if __name__ == "__main__":
    add_flips()
