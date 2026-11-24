"""Hinglish system: character TF-IDF, a negation-scoped lexicon, and a stacker.

The lexicon and the character model are fit on the official training split.
The stacker and the neutral margin are fit on the validation split only.
The test split is scored once. HingBERT is not reloaded here, so this can run
while a fine-tune is using the GPU. Disagreement with the saved HingBERT
predictions is reported when that file already exists.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

try:
    from src.data import ID2LABEL
    from src.lexicon import (
        build_lexicon,
        has_elongation,
        has_negation,
        score_text,
    )
    from src.models import build_tfidf_baseline
    from src.preprocess import clean_tweet
except ImportError:
    from data import ID2LABEL
    from lexicon import build_lexicon, has_elongation, has_negation, score_text
    from models import build_tfidf_baseline
    from preprocess import clean_tweet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SYSTEM = os.path.join(RESULTS, "system")
MARGINS = (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)


def _clean_frame(frame):
    out = frame.copy()
    out["clean_text"] = [clean_tweet(text) for text in out["text"].astype(str)]
    out["label"] = out["label"].astype(int)
    return out


def _lexicon_matrix(texts, weights):
    rows = []
    triggers = []
    for text in texts:
        polarity, pos_mass, neg_mass, trigger = score_text(text, weights)
        rows.append([polarity, pos_mass, neg_mass, 1.0 if has_negation(text) else 0.0])
        triggers.append(trigger)
    return np.asarray(rows, dtype=np.float64), triggers


def _margin(proba):
    order = np.sort(proba, axis=1)
    return order[:, -1] - order[:, -2]


def _apply_margin(proba, threshold):
    pred = proba.argmax(axis=1)
    if threshold <= 0:
        return pred
    close = _margin(proba) < threshold
    pred = pred.copy()
    pred[close] = 1
    return pred


def _specialist(tfidf_label, lex_label, final_label, tfidf_margin, lex_margin):
    close = max(tfidf_margin, lex_margin) < 0.12
    if final_label == "neutral" and close and tfidf_label != "neutral":
        return "neutral gate"
    if final_label == lex_label and final_label != tfidf_label:
        return "scoped lexicon"
    if final_label == tfidf_label and final_label != lex_label:
        return "character model"
    return "both specialists"


class HinglishSystem:
    def __init__(self, weights, lex_clf, tfidf, router, margin):
        self.weights = weights
        self.lex_clf = lex_clf
        self.tfidf = tfidf
        self.router = router
        self.margin = float(margin)

    def _parts(self, cleaned):
        lex_x, triggers = _lexicon_matrix([cleaned], self.weights)
        lex_proba = self.lex_clf.predict_proba(lex_x)
        lex_proba = _align_proba(lex_proba, self.lex_clf.classes_)
        tfidf_proba = self.tfidf.predict_proba([cleaned])
        tfidf_proba = _align_proba(tfidf_proba, self.tfidf.named_steps["classifier"].classes_)
        return lex_proba[0], tfidf_proba[0], triggers[0]

    def explain(self, text):
        cleaned = clean_tweet(str(text or ""))
        if not cleaned:
            return {
                "label": "neutral",
                "confidence": 0.3333,
                "probabilities": {"negative": 0.3333, "neutral": 0.3334, "positive": 0.3333},
                "cleaned_text": "",
                "gap": 0.0,
                "trigger": "",
                "specialist": "neutral gate",
                "reading": "Empty input. The call is neutral.",
            }
        lex_p, tfidf_p, trigger = self._parts(cleaned)
        features = _stack_row(tfidf_p, lex_p, cleaned).reshape(1, -1)
        proba = _align_proba(self.router.predict_proba(features), self.router.classes_)[0]
        pred = int(_apply_margin(proba.reshape(1, -1), self.margin)[0])
        label = ID2LABEL[pred]
        gap = float(_margin(proba.reshape(1, -1))[0])
        tfidf_label = ID2LABEL[int(np.argmax(tfidf_p))]
        lex_label = ID2LABEL[int(np.argmax(lex_p))]
        specialist = _specialist(
            tfidf_label, lex_label, label, float(_margin(tfidf_p.reshape(1, -1))[0]),
            float(_margin(lex_p.reshape(1, -1))[0]),
        )
        word = trigger or "none"
        reading = (
            f"Gap {gap:.2f}. Specialist: {specialist}. Trigger: {word}. "
            f"Character model says {tfidf_label}. Scoped lexicon says {lex_label}."
        )
        return {
            "label": label,
            "confidence": round(float(proba[pred]), 4),
            "probabilities": {ID2LABEL[i]: round(float(proba[i]), 4) for i in range(3)},
            "cleaned_text": cleaned,
            "gap": round(gap, 4),
            "trigger": trigger,
            "specialist": specialist,
            "reading": reading,
        }


def _align_proba(proba, classes):
    out = np.zeros((len(proba), 3), dtype=np.float64)
    for column, label in enumerate(classes):
        out[:, int(label)] = proba[:, column]
    return out


def _stack_row(tfidf_p, lex_p, text):
    tfidf_m = float(_margin(tfidf_p.reshape(1, -1))[0])
    lex_m = float(_margin(lex_p.reshape(1, -1))[0])
    disagree = float(int(np.argmax(tfidf_p)) != int(np.argmax(lex_p)))
    return np.asarray([
        tfidf_p[0], tfidf_p[1], tfidf_p[2],
        lex_p[0], lex_p[1], lex_p[2],
        tfidf_m, lex_m, min(tfidf_m, lex_m), disagree,
        1.0 if has_negation(text) else 0.0,
        1.0 if has_elongation(text) else 0.0,
    ], dtype=np.float64)


def _stack_matrix(tfidf_p, lex_p, texts):
    return np.vstack([
        _stack_row(tfidf_p[i], lex_p[i], texts[i]) for i in range(len(texts))
    ])


def _scores(name, gold, pred):
    per = f1_score(gold, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "model": name,
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "f1_negative": float(per[0]),
        "f1_neutral": float(per[1]),
        "f1_positive": float(per[2]),
    }


def _band(cmi):
    value = float(cmi)
    if value < 15:
        return "mostly one language"
    return "mixed"


def _write_slices(test_df, pred_map):
    frame = test_df.copy()
    frame["band"] = frame["cmi"].map(_band) if "cmi" in frame.columns else "all"
    frame["negation_band"] = [
        "has negation" if has_negation(text) else "no negation" for text in frame["clean_text"]
    ]
    rows = []
    for column in ("band", "negation_band"):
        for name, group in frame.groupby(column):
            idx = group.index.to_numpy()
            for model, pred in pred_map.items():
                scores = _scores(model, frame.loc[idx, "label"].to_numpy(), pred[idx])
                scores["slice"] = f"{column}={name}"
                scores["n"] = int(len(group))
                rows.append(scores)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "slices.csv"), index=False)
    return out


def _write_disagreements(test_df, tfidf_pred, system_pred):
    hing_path = os.path.join(RESULTS, "preds", "hingbert.csv")
    if not os.path.exists(hing_path):
        print("[System] No saved HingBERT predictions yet. Disagreement table skipped.")
        return
    hing = pd.read_csv(hing_path)
    if len(hing) != len(test_df):
        print("[System] HingBERT prediction file length does not match the test set.")
        return
    gold = test_df["label"].to_numpy()
    hing_pred = hing["pred"].astype(int).to_numpy()
    rows = []
    for i in range(len(test_df)):
        if int(tfidf_pred[i]) == int(hing_pred[i]):
            continue
        text = str(test_df.iloc[i]["text"])
        tfidf_ok = int(tfidf_pred[i]) == int(gold[i])
        hing_ok = int(hing_pred[i]) == int(gold[i])
        if tfidf_ok and not hing_ok:
            winner = "character model"
        elif hing_ok and not tfidf_ok:
            winner = "hingbert"
        elif tfidf_ok and hing_ok:
            winner = "both"
        else:
            winner = "neither"
        rows.append({
            "text": text,
            "gold": ID2LABEL[int(gold[i])],
            "character_model": ID2LABEL[int(tfidf_pred[i])],
            "hingbert": ID2LABEL[int(hing_pred[i])],
            "system": ID2LABEL[int(system_pred[i])],
            "winner": winner,
            "negation": int(has_negation(text)),
            "elongation": int(has_elongation(text)),
        })
    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(RESULTS, "disagreements.csv"), index=False)
    summary_rows = []
    if len(detail):
        for flag, title in ((None, "all disagreements"), ("negation", "negation disagreements"), ("elongation", "elongation disagreements")):
            part = detail if flag is None else detail[detail[flag] == 1]
            if part.empty:
                continue
            counts = part["winner"].value_counts().to_dict()
            summary_rows.append({
                "slice": title,
                "n": int(len(part)),
                "character_model_right": int(counts.get("character model", 0)),
                "hingbert_right": int(counts.get("hingbert", 0)),
                "neither": int(counts.get("neither", 0)),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS, "disagreement_summary.csv"), index=False)
    print("[System] Disagreements")
    print(summary.to_string(index=False))


def fit_and_save():
    train_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_train.csv")))
    val_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_val.csv")))
    test_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_test.csv")))

    weights, table = build_lexicon(train_df["clean_text"], train_df["label"])
    lex_train, _ = _lexicon_matrix(train_df["clean_text"], weights)
    lex_clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lex_clf.fit(lex_train, train_df["label"])

    tfidf = build_tfidf_baseline("logreg", use_class_weights=True)
    tfidf.fit(train_df["clean_text"], train_df["label"])

    def specialist_proba(frame):
        lex_x, _ = _lexicon_matrix(frame["clean_text"], weights)
        lex_p = _align_proba(lex_clf.predict_proba(lex_x), lex_clf.classes_)
        tfidf_p = _align_proba(
            tfidf.predict_proba(frame["clean_text"]),
            tfidf.named_steps["classifier"].classes_,
        )
        return tfidf_p, lex_p

    val_tfidf, val_lex = specialist_proba(val_df)
    test_tfidf, test_lex = specialist_proba(test_df)
    x_val = _stack_matrix(val_tfidf, val_lex, val_df["clean_text"])
    x_test = _stack_matrix(test_tfidf, test_lex, test_df["clean_text"])
    router = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    router.fit(x_val, val_df["label"])
    val_proba = _align_proba(router.predict_proba(x_val), router.classes_)

    best_t = 0.0
    best_f1 = -1.0
    for threshold in MARGINS:
        pred = _apply_margin(val_proba, threshold)
        score = f1_score(val_df["label"], pred, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(threshold)

    test_proba = _align_proba(router.predict_proba(x_test), router.classes_)
    system_pred = _apply_margin(test_proba, best_t)
    tfidf_pred = test_tfidf.argmax(axis=1)
    lex_pred = test_lex.argmax(axis=1)
    gold = test_df["label"].to_numpy()
    metrics = pd.DataFrame([
        _scores("character_tfidf", gold, tfidf_pred),
        _scores("scoped_lexicon", gold, lex_pred),
        _scores("hinglish_system", gold, system_pred),
    ])
    metrics.to_csv(os.path.join(RESULTS, "system_metrics.csv"), index=False)

    os.makedirs(SYSTEM, exist_ok=True)
    joblib.dump(tfidf, os.path.join(SYSTEM, "tfidf.joblib"))
    joblib.dump(lex_clf, os.path.join(SYSTEM, "lexicon_clf.joblib"))
    joblib.dump(router, os.path.join(SYSTEM, "router.joblib"))
    with open(os.path.join(SYSTEM, "lexicon.json"), "w", encoding="utf-8") as handle:
        json.dump(weights, handle)
    pd.DataFrame(table).head(80).to_csv(os.path.join(SYSTEM, "lexicon_top.csv"), index=False)
    meta = {
        "lexicon_words": len(weights),
        "neutral_margin": best_t,
        "val_macro_f1_at_margin": best_f1,
        "fit_on": "character model and lexicon on train; stacker and margin on validation",
    }
    with open(os.path.join(SYSTEM, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    pred_map = {
        "character_tfidf": tfidf_pred,
        "scoped_lexicon": lex_pred,
        "hinglish_system": system_pred,
    }
    hing_path = os.path.join(RESULTS, "preds", "hingbert.csv")
    if os.path.exists(hing_path):
        hing = pd.read_csv(hing_path)
        if len(hing) == len(test_df):
            pred_map["hingbert"] = hing["pred"].astype(int).to_numpy()
    _write_slices(test_df, pred_map)
    _write_disagreements(test_df, tfidf_pred, system_pred)
    print("[System] Test metrics")
    print(metrics.to_string(index=False))
    print(f"[System] Lexicon words={len(weights)} neutral margin={best_t} val macro-F1={best_f1:.4f}")
    return metrics


def load_system(directory=None):
    directory = directory or SYSTEM
    router_path = os.path.join(directory, "router.joblib")
    if not os.path.exists(router_path):
        return None
    with open(os.path.join(directory, "lexicon.json"), encoding="utf-8") as handle:
        weights = json.load(handle)
    meta_path = os.path.join(directory, "meta.json")
    margin = 0.0
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            margin = json.load(handle).get("neutral_margin", 0.0)
    return HinglishSystem(
        weights=weights,
        lex_clf=joblib.load(os.path.join(directory, "lexicon_clf.joblib")),
        tfidf=joblib.load(os.path.join(directory, "tfidf.joblib")),
        router=joblib.load(router_path),
        margin=margin,
    )


if __name__ == "__main__":
    fit_and_save()
