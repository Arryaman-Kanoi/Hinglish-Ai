"""Negation and contrast scope for Hinglish sentiment.

SentiMix has no sarcasm labels, so this module does not train a sarcasm class.
A negation flips the content word two tokens before it and the next three
tokens. A contrast word (lekin, but, magar, and the same set below) lets the
later clause override the earlier one. A sentence-final "na" is a tag
("right?"), not a second negation. With a negated word in front of it, that
tag is the echo shape in "main achha nahi hoon na": the line throws the other
person's words back. The earlier turn is not in the tweet, so the echo is a
sentence shape, not a sarcasm class. Those scope features are added to the
saved character model and lexicon. The logistic layer is fit on 70% of the
validation split. The other 30% decides whether it is kept. The test split is
scored once, after that decision.

Praise words plus a laugh or clap emoji are a text cue only. The cue can
receive weight toward a sentiment class. It is not a sarcasm label.
"""

import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from src.data import ID2LABEL
    from src.lexicon import NEGATION, SKIP, score_text, tokens_of
    from src.system import (
        DATA,
        RESULTS,
        SYSTEM,
        _align_proba,
        _apply_margin,
        _clean_frame,
        _lexicon_matrix,
        _margin,
        _scores,
        _stack_matrix,
        _stack_row,
        load_system,
    )
except ImportError:
    from data import ID2LABEL
    from lexicon import NEGATION, SKIP, score_text, tokens_of
    from system import (
        DATA,
        RESULTS,
        SYSTEM,
        _align_proba,
        _apply_margin,
        _clean_frame,
        _lexicon_matrix,
        _margin,
        _scores,
        _stack_matrix,
        _stack_row,
        load_system,
    )

CONTRAST = frozenset({
    "lekin", "lekhin", "magar", "magr", "parantu", "balki",
    "but", "however", "although", "though", "warna",
})
INTENSIFIERS = frozenset({
    "wah", "waah", "kya", "bohot", "boht", "bahut", "bahot",
    "itna", "itni", "kitna", "bilkul",
})
PRAISE_CUES = frozenset({"dhanyawad", "shukriya", "shukria"})
EMOJI_CUES = ("😂", "👏", "🤣", "😅", "🙄")
FORWARD = 3
BACKWARD = 2
SURFACE_EPS = 0.25
HOLDOUT_FRACTION = 0.30
MIN_SLICE = 40
# Kept only if holdout overall macro-F1 stays within this of the current
# system and a flip slice gains at least this much macro-F1.
OVERALL_SLACK = 0.005
SLICE_GAIN = 0.01

EXTRA_NAMES = [
    "surface_polarity",
    "scoped_polarity",
    "scope_conflict",
    "has_contrast",
    "first_clause_polarity",
    "last_clause_polarity",
    "contrast_override",
    "praise_mass",
    "emoji_cue",
    "sarcasm_cue",
    "praise_no_scope",
    "cue_and_praise",
    "tag_na",
    "echo_shape",
]
TAG_TOKENS = frozenset({"na", "naa", "nna", "haina"})
# Auxiliaries and pronouns are not sentiment. "hu" in "acha nahi hu na" must not
# become the flipped word.
CLOSED = SKIP | {
    "hu", "hoon", "hun", "ho", "h", "hai", "hain", "tha", "thi", "the",
    "main", "me", "hum", "tum", "aap", "mujhe", "mera", "meri", "tere", "teri",
}
# Frequent polarity words the mined lexicon can miss because "achha nahi"
# mixes them into both classes. Counts below use train tweets with no
# real negation, so the word's own sign is not poisoned by the flip.
SCOPE_WORDS = (
    "acha", "accha", "achha", "acchi", "acche", "achhe", "achhi",
    "bura", "buri", "bure", "kharab", "kharab", "bekar", "bekaar",
    "bakwas", "bakwaas", "pasand", "pyar", "pyaar", "ganda", "gandi",
    "mast", "zabardast", "badhiya", "badiya", "bekar",
)


def _sign(value, eps=SURFACE_EPS):
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _flips(tokens, weights):
    """Mark lexicon tokens whose sign a nearby negation reverses."""
    flip = [False] * len(tokens)
    nearest = [""] * len(tokens)
    for i, tok in enumerate(tokens):
        if tok not in NEGATION:
            continue
        seen = 0
        j = i - 1
        while j >= 0 and seen < BACKWARD:
            if tokens[j] not in NEGATION and tokens[j] not in CLOSED:
                seen += 1
                if weights.get(tokens[j], 0.0) != 0.0:
                    flip[j] = True
                    nearest[j] = tok
            j -= 1
        left = FORWARD
        j = i + 1
        while j < len(tokens) and left > 0:
            if tokens[j] in NEGATION or tokens[j] in CLOSED:
                j += 1
                continue
            if weights.get(tokens[j], 0.0) != 0.0:
                flip[j] = True
                nearest[j] = tok
            left -= 1
            j += 1
    return flip, nearest


def _polarity(tokens, weights, flip=None, nearest=None):
    polarity = 0.0
    praise = 0.0
    trigger = ""
    trigger_abs = 0.0
    flip = flip or [False] * len(tokens)
    nearest = nearest or [""] * len(tokens)
    for i, tok in enumerate(tokens):
        weight = weights.get(tok, 0.0)
        if weight == 0.0 or tok in NEGATION or tok in CLOSED:
            continue
        signed = -weight if flip[i] else weight
        polarity += signed
        if signed > 0 and not flip[i]:
            praise += signed
        if abs(signed) >= trigger_abs:
            trigger_abs = abs(signed)
            marker = nearest[i]
            trigger = f"{marker} {tok}".strip() if marker else tok
    return polarity, praise, trigger


def _strip_tag(tokens):
    """A final na/haina asks 'right?'. It does not negate the clause."""
    if tokens and tokens[-1] in TAG_TOKENS:
        return tokens[:-1], True
    return tokens, False


def _supplement(texts, labels, weights):
    """Add train-only signs for frequent words the mined lexicon skipped."""
    added = {}
    real_neg = NEGATION - TAG_TOKENS
    wanted = [word for word in dict.fromkeys(SCOPE_WORDS) if word not in weights]
    counts = {word: [0, 0] for word in wanted}
    for text, label in zip(texts, labels):
        label = int(label)
        if label not in (0, 2):
            continue
        toks = tokens_of(text)
        body, _ = _strip_tag(toks)
        if any(tok in real_neg for tok in body):
            continue
        seen = set(body)
        for word in wanted:
            if word in seen:
                counts[word][0 if label == 2 else 1] += 1
    for word, (pos, neg) in counts.items():
        if pos + neg < 15:
            continue
        polarity = math.log((pos + 1) / (neg + 1))
        if abs(polarity) < 0.25:
            continue
        added[word] = float(polarity)
    return added


def analyze(text, weights):
    """Text-only scope. Gold labels are not an input."""
    raw = str(text or "")
    tokens, tag = _strip_tag(tokens_of(raw))
    surface, praise_plain, _ = _polarity(tokens, weights)
    flip, nearest = _flips(tokens, weights)
    scoped, praise, trigger = _polarity(tokens, weights, flip, nearest)
    parts = []
    current = []
    has_contrast = 0.0
    for tok in tokens:
        if tok in CONTRAST:
            has_contrast = 1.0
            if current:
                parts.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        parts.append(current)
    first_p = 0.0
    last_p = scoped
    if has_contrast and parts:
        first_tokens = parts[0]
        last_tokens = parts[-1] if len(parts) > 1 else []
        first_flip, first_near = _flips(first_tokens, weights)
        last_flip, last_near = _flips(last_tokens, weights)
        first_p, _, _ = _polarity(first_tokens, weights, first_flip, first_near)
        if last_tokens:
            last_p, _, last_trigger = _polarity(last_tokens, weights, last_flip, last_near)
            if last_trigger:
                trigger = last_trigger
        else:
            last_p = 0.0
    first_sign = _sign(first_p)
    last_sign = _sign(last_p)
    override = 1.0 if has_contrast and first_sign and last_sign and first_sign != last_sign else 0.0
    surface_sign = _sign(surface)
    scoped_sign = _sign(scoped)
    conflict = 1.0 if surface_sign and scoped_sign and surface_sign != scoped_sign else 0.0
    emoji = 1.0 if any(mark in raw for mark in EMOJI_CUES) else 0.0
    cue_word = 1.0 if any(tok in PRAISE_CUES for tok in tokens) else 0.0
    sarcasm_cue = 1.0 if emoji or cue_word else 0.0
    negated = 1.0 if any(tok in NEGATION for tok in tokens) else 0.0
    praise_no_scope = praise if not negated and not has_contrast else 0.0
    cue_and_praise = sarcasm_cue if praise > SURFACE_EPS else 0.0
    tag_na = 1.0 if tag else 0.0
    echo = 1.0 if tag and any(flip) else 0.0
    if echo and not override:
        kind = "echo"
    elif override:
        kind = "contrast"
    elif conflict or (negated and any(flip)):
        kind = "negation"
    elif tag:
        kind = "tag"
    elif sarcasm_cue and not negated and not has_contrast:
        kind = "praise cue"
    else:
        kind = "none"
    intensifier = 0.0
    for i, tok in enumerate(tokens):
        if tok not in INTENSIFIERS:
            continue
        window = tokens[max(0, i - 2): i + 3]
        if any(weights.get(item, 0.0) > 0 for item in window):
            intensifier = 1.0
            break
    return {
        "surface": surface,
        "scoped": scoped,
        "conflict": conflict,
        "has_negation": negated,
        "has_contrast": has_contrast,
        "first_p": first_p,
        "last_p": last_p,
        "override": override,
        "praise": praise,
        "emoji": emoji,
        "sarcasm_cue": sarcasm_cue,
        "praise_no_scope": praise_no_scope,
        "cue_and_praise": cue_and_praise,
        "tag_na": tag_na,
        "echo_shape": echo,
        "intensifier": intensifier,
        "kind": kind,
        "trigger": trigger,
        "surface_sign": surface_sign,
        "scoped_sign": scoped_sign,
        "last_sign": last_sign,
    }


def scope_sentence(scope):
    trigger = scope.get("trigger") or "the content word"
    kind = scope["kind"]
    if kind == "echo":
        return (
            f"Scope: echo. The negation flips {trigger}, and the final na means 'right?'. "
            "Written this way, the line throws someone's words back at them. "
            "The earlier turn is not in the tweet, so this is not marked as sarcasm. "
            "The sentiment follows the flipped word."
        )
    if kind == "tag":
        return "Scope: the final na is a tag, 'right?', not a negation. Nothing else flips the polarity."
    if kind == "contrast":
        return (
            f"Scope: contrast. The clause after lekin/but/magar overrides the earlier clause. "
            f"Clause trigger: {trigger}."
        )
    if kind == "negation":
        return (
            f"Scope: negation. The negation flips the content word beside it "
            f"(two tokens back, three tokens forward). Flip: {trigger}."
        )
    if kind == "praise cue":
        return (
            "Scope: praise cue, and there is no negation or contrast. "
            "SentiMix has no sarcasm label, so this line is not marked sarcastic."
        )
    return "Scope: no polarity flip."


def feature_row(text, weights, tfidf_p, lex_p, scope=None):
    scope = scope or analyze(text, weights)
    base = _stack_row(tfidf_p, lex_p, text)
    return np.concatenate([base, _extra_vector(scope)]), scope


def _extra_vector(scope):
    return np.asarray([
        scope["surface"],
        scope["scoped"],
        scope["conflict"],
        scope["has_contrast"],
        scope["first_p"],
        scope["last_p"],
        scope["override"],
        scope["praise"],
        scope["emoji"],
        scope["sarcasm_cue"],
        scope["praise_no_scope"],
        scope["cue_and_praise"],
        scope["tag_na"],
        scope["echo_shape"],
    ], dtype=np.float64)


def feature_matrix(texts, weights, tfidf_p, lex_p):
    rows = []
    scopes = []
    for i, text in enumerate(texts):
        row, scope = feature_row(text, weights, tfidf_p[i], lex_p[i])
        rows.append(row)
        scopes.append(scope)
    return np.vstack(rows), scopes


def _specialist_proba(system, texts):
    lex_x, _ = _lexicon_matrix(texts, system.weights)
    lex_p = _align_proba(system.lex_clf.predict_proba(lex_x), system.lex_clf.classes_)
    tfidf_p = _align_proba(
        system.tfidf.predict_proba(list(texts)),
        system.tfidf.named_steps["classifier"].classes_,
    )
    return tfidf_p, lex_p


def _system_pred(system, tfidf_p, lex_p, texts):
    stacked = _stack_matrix(tfidf_p, lex_p, texts)
    proba = _align_proba(system.router.predict_proba(stacked), system.router.classes_)
    return _apply_margin(proba, system.margin)


def _predict_flip(router, scaler, features):
    proba = _align_proba(router.predict_proba(scaler.transform(features)), router.classes_)
    return proba.argmax(axis=1), proba


def _slice_names(scopes):
    negation = np.asarray([scope["has_negation"] == 1.0 for scope in scopes])
    contrast = np.asarray([scope["has_contrast"] == 1.0 for scope in scopes])
    conflict = np.asarray([scope["conflict"] == 1.0 for scope in scopes])
    praise = np.asarray([
        scope["praise"] > SURFACE_EPS and scope["has_negation"] == 0.0 and scope["has_contrast"] == 0.0
        for scope in scopes
    ])
    echo = np.asarray([scope["echo_shape"] == 1.0 for scope in scopes])
    return {
        "all": np.ones(len(scopes), dtype=bool),
        "negation": negation,
        "contrast": contrast,
        "scope_conflict": conflict,
        "praise_no_flip": praise,
        "echo": echo,
    }


def _metric_rows(split, pred_map, gold, scopes):
    rows = []
    slices = _slice_names(scopes)
    for name, pred in pred_map.items():
        pred = np.asarray(pred)
        for slice_name, mask in slices.items():
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                continue
            scores = _scores(name, gold[idx], pred[idx])
            scores["split"] = split
            scores["slice"] = slice_name
            scores["n"] = int(len(idx))
            scores["gold_negative_rate"] = float(np.mean(gold[idx] == 0))
            rows.append(scores)
    return rows


def _lookup(rows, split, model, slice_name):
    for row in rows:
        if row["split"] == split and row["model"] == model and row["slice"] == slice_name:
            return row
    return None


def _gain(rows, slice_name):
    flip = _lookup(rows, "holdout", "flip", slice_name)
    base = _lookup(rows, "holdout", "hinglish_system", slice_name)
    if flip is None or base is None or flip["n"] < MIN_SLICE:
        return None
    return float(flip["macro_f1"] - base["macro_f1"])


def _self_check(weights):
    negated = analyze("yeh movie awesome nahi hai", weights)
    forward = analyze("nahi awesome movie", weights)
    plain = analyze("yeh movie awesome hai", weights)
    contrast = analyze("movie awesome hai but ending shame hai", weights)
    if negated["kind"] != "negation":
        raise RuntimeError(f"backward negation was not a flip: {negated}")
    if forward["kind"] != "negation":
        raise RuntimeError(f"forward negation was not a flip: {forward}")
    if plain["kind"] == "negation":
        raise RuntimeError(f"plain praise was marked as negation: {plain}")
    if contrast["kind"] != "contrast":
        raise RuntimeError(f"contrast clause was not marked: {contrast}")
    # Hindi order: the positive word sits before nahi, so the forward-only
    # window leaves it positive. The backward window has to reverse it.
    forward_only, _, _, _ = score_text("yeh movie awesome nahi hai", weights, window=FORWARD)
    if not (forward_only > 0 and negated["scoped"] < 0):
        raise RuntimeError(
            f"backward window did not change the Hindi order: forward={forward_only} scoped={negated['scoped']}"
        )
    echo = analyze("me awesome nahi hu na", weights)
    tag = analyze("yeh movie awesome hai na", weights)
    if echo["kind"] != "echo" or echo["scoped"] >= 0:
        raise RuntimeError(f"echo shape was not negative: {echo}")
    if tag["kind"] != "tag" or tag["scoped"] <= 0:
        raise RuntimeError(f"final na was treated as negation: {tag}")


def _census_row(split, frame, weights):
    scopes = [analyze(text, weights) for text in frame["clean_text"]]
    gold = frame["label"].to_numpy()
    surface_disagree = 0
    explained = 0
    residual = 0
    for scope, label in zip(scopes, gold):
        gold_sign = -1 if label == 0 else (1 if label == 2 else 0)
        if scope["surface_sign"] == 0 or gold_sign == 0 or scope["surface_sign"] == gold_sign:
            continue
        surface_disagree += 1
        fixed = scope["scoped_sign"] == gold_sign or (scope["has_contrast"] and scope["last_sign"] == gold_sign)
        if fixed:
            explained += 1
        else:
            residual += 1
    return {
        "split": split,
        "n": int(len(frame)),
        "negation": int(sum(scope["has_negation"] for scope in scopes)),
        "contrast": int(sum(scope["has_contrast"] for scope in scopes)),
        "scope_conflict": int(sum(scope["conflict"] for scope in scopes)),
        "surface_disagrees_with_gold": surface_disagree,
        "scope_explains_disagreement": explained,
        "residual_unexplained": residual,
        "praise_no_flip": int(sum(
            scope["praise"] > SURFACE_EPS and not scope["has_negation"] and not scope["has_contrast"]
            for scope in scopes
        )),
        "praise_cue": int(sum(scope["kind"] == "praise cue" for scope in scopes)),
        "echo": int(sum(scope["echo_shape"] for scope in scopes)),
        "tag_na": int(sum(scope["tag_na"] for scope in scopes)),
    }


def _examples(test_df, gold, pred_map, scopes, limit=12):
    blocked = {
        "gandu", "suwar", "chutiye", "chutiya", "chutiyaa", "harami", "haramkhor",
        "tatti", "bhosdi", "madarchod", "behenchod", "randi",
    }
    rows = []
    order = {"contrast": 0, "negation": 1, "praise cue": 2}
    ranked = sorted(range(len(scopes)), key=lambda i: order.get(scopes[i]["kind"], 9))
    for i in ranked:
        scope = scopes[i]
        if scope["kind"] == "none":
            continue
        text = str(test_df["text"].iloc[i])
        if blocked.intersection(tokens_of(text)):
            continue
        row = {
            "text": text,
            "gold": ID2LABEL[int(gold[i])],
            "kind": scope["kind"],
            "trigger": scope["trigger"],
            "character_tfidf": ID2LABEL[int(pred_map["character_tfidf"][i])],
            "hinglish_system": ID2LABEL[int(pred_map["hinglish_system"][i])],
            "flip": ID2LABEL[int(pred_map["flip"][i])],
        }
        if "hingbert" in pred_map:
            row["hingbert"] = ID2LABEL[int(pred_map["hingbert"][i])]
        rows.append(row)
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


_BUNDLE = None


def load_bundle():
    """Return the kept scope layer, or None when the holdout rejected it."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE if _BUNDLE.get("kept") else None
    path = os.path.join(SYSTEM, "flip_meta.json")
    router_path = os.path.join(SYSTEM, "flip_router.joblib")
    if not os.path.exists(path) or not os.path.exists(router_path):
        _BUNDLE = {"kept": False}
        return None
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if not meta.get("kept"):
        _BUNDLE = {"kept": False}
        return None
    import joblib
    _BUNDLE = {
        "kept": True,
        "router": joblib.load(router_path),
        "scaler": joblib.load(os.path.join(SYSTEM, "flip_scaler.joblib")),
    }
    return _BUNDLE


def annotate(result, text, system):
    """Add the scope sentence. Replace the label only when the layer was kept."""
    cleaned = result.get("cleaned_text") or ""
    if not cleaned:
        out = dict(result)
        out["flip"] = "none"
        return out
    scope = analyze(cleaned, scope_weights(system.weights))
    out = dict(result)
    out["flip"] = scope["kind"]
    out["reading"] = f"{(result.get('reading') or '').rstrip()} {scope_sentence(scope)}".strip()
    bundle = load_bundle()
    if bundle is None:
        return out
    lex_p, tfidf_p, _ = system._parts(cleaned)
    row, _ = feature_row(cleaned, scope_weights(system.weights), tfidf_p, lex_p, scope)
    pred, proba = _predict_flip(bundle["router"], bundle["scaler"], row.reshape(1, -1))
    pred_id = int(pred[0])
    out["label"] = ID2LABEL[pred_id]
    out["confidence"] = round(float(proba[0, pred_id]), 4)
    out["probabilities"] = {ID2LABEL[i]: round(float(proba[0, i]), 4) for i in range(3)}
    out["gap"] = round(float(_margin(proba)), 4)
    if out["label"] != result.get("label"):
        out["specialist"] = "scope layer"
    return out


def scope_weights(base):
    merged = dict(base)
    path = os.path.join(SYSTEM, "scope_extra.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            merged.update(json.load(handle))
    return merged


def fit_and_score():
    system = load_system()
    if system is None:
        raise RuntimeError("results/system/router.joblib is missing. Run python -m src.system first.")

    train_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_train.csv")))
    val_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_val.csv")))
    test_df = _clean_frame(pd.read_csv(os.path.join(DATA, "sentimix_test.csv")))
    added = _supplement(train_df["clean_text"], train_df["label"], system.weights)
    os.makedirs(SYSTEM, exist_ok=True)
    with open(os.path.join(SYSTEM, "scope_extra.json"), "w", encoding="utf-8") as handle:
        json.dump(added, handle, indent=2)
    weights = scope_weights(system.weights)
    _self_check(weights)
    census = pd.DataFrame([
        _census_row("train", train_df, weights),
        _census_row("val", val_df, weights),
        _census_row("test", test_df, weights),
    ])
    census.to_csv(os.path.join(RESULTS, "flip_census.csv"), index=False)

    val_texts = val_df["clean_text"].tolist()
    test_texts = test_df["clean_text"].tolist()
    val_tfidf, val_lex = _specialist_proba(system, val_texts)
    test_tfidf, test_lex = _specialist_proba(system, test_texts)
    val_x, val_scopes = feature_matrix(val_texts, weights, val_tfidf, val_lex)
    test_x, test_scopes = feature_matrix(test_texts, weights, test_tfidf, test_lex)

    rng = np.random.RandomState(42)
    order = rng.permutation(len(val_df))
    hold_n = int(round(len(val_df) * HOLDOUT_FRACTION))
    hold_idx = np.sort(order[:hold_n])
    fit_idx = np.sort(order[hold_n:])
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(val_x[fit_idx])
    router = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    router.fit(x_fit, val_df["label"].to_numpy()[fit_idx])

    hold_pred, _ = _predict_flip(router, scaler, val_x[hold_idx])
    hold_gold = val_df["label"].to_numpy()[hold_idx]
    hold_scopes = [val_scopes[i] for i in hold_idx]
    hold_tfidf = val_tfidf[hold_idx].argmax(axis=1)
    hold_lex = val_lex[hold_idx].argmax(axis=1)
    hold_system = _system_pred(system, val_tfidf[hold_idx], val_lex[hold_idx], [val_texts[i] for i in hold_idx])
    rows = _metric_rows("holdout", {
        "character_tfidf": hold_tfidf,
        "scoped_lexicon": hold_lex,
        "hinglish_system": hold_system,
        "flip": hold_pred,
    }, hold_gold, hold_scopes)

    neg_gain = _gain(rows, "negation")
    contrast_gain = _gain(rows, "contrast")
    echo_gain = _gain(rows, "echo")
    conflict_gain = _gain(rows, "scope_conflict")
    flip_all = _lookup(rows, "holdout", "flip", "all")
    system_all = _lookup(rows, "holdout", "hinglish_system", "all")
    slice_gains = [gain for gain in (neg_gain, contrast_gain, echo_gain) if gain is not None]
    kept = bool(
        flip_all
        and system_all
        and flip_all["macro_f1"] + OVERALL_SLACK >= system_all["macro_f1"]
        and any(gain >= SLICE_GAIN for gain in slice_gains)
    )

    # One test look. The kept demo layer is refit on all validation rows.
    # Refit does not see the test labels. The holdout already made the decision.
    report_router = router
    report_scaler = scaler
    if kept:
        report_scaler = StandardScaler()
        x_all = report_scaler.fit_transform(val_x)
        report_router = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        report_router.fit(x_all, val_df["label"].to_numpy())
        import joblib
        os.makedirs(SYSTEM, exist_ok=True)
        joblib.dump(report_router, os.path.join(SYSTEM, "flip_router.joblib"))
        joblib.dump(report_scaler, os.path.join(SYSTEM, "flip_scaler.joblib"))

    test_pred, _ = _predict_flip(report_router, report_scaler, test_x)
    test_gold = test_df["label"].to_numpy()
    pred_map = {
        "character_tfidf": test_tfidf.argmax(axis=1),
        "scoped_lexicon": test_lex.argmax(axis=1),
        "hinglish_system": _system_pred(system, test_tfidf, test_lex, test_texts),
        "flip": test_pred,
    }
    hing_path = os.path.join(RESULTS, "preds", "hingbert.csv")
    if os.path.exists(hing_path):
        hing = pd.read_csv(hing_path)
        same = len(hing) == len(test_df) and hing["text"].astype(str).tolist() == test_df["text"].astype(str).tolist()
        if same:
            pred_map["hingbert"] = hing["pred"].astype(int).to_numpy()
    rows.extend(_metric_rows("test", pred_map, test_gold, test_scopes))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(RESULTS, "flip_metrics.csv"), index=False)
    _examples(test_df, test_gold, pred_map, test_scopes).to_csv(
        os.path.join(RESULTS, "flip_examples.csv"), index=False
    )

    coefficients = report_router.coef_
    classes = [int(label) for label in report_router.classes_]
    # Base stack has 12 columns, then EXTRA_NAMES. Names are for the report.
    base_names = [
        "tfidf_negative", "tfidf_neutral", "tfidf_positive",
        "lex_negative", "lex_neutral", "lex_positive",
        "tfidf_margin", "lex_margin", "min_margin", "specialist_disagree",
        "has_negation", "has_elongation",
    ]
    names = base_names + EXTRA_NAMES
    coef_rows = []
    for class_index, label in enumerate(classes):
        for feature_index, name in enumerate(names):
            coef_rows.append({
                "class": ID2LABEL[label],
                "feature": name,
                "coefficient": float(coefficients[class_index, feature_index]),
            })
    pd.DataFrame(coef_rows).to_csv(os.path.join(RESULTS, "flip_coefficients.csv"), index=False)

    decision = {
        "kept": kept,
        "reason": (
            "Holdout macro-F1 stayed near the current system and a negation or contrast slice gained at least 0.01."
            if kept else
            "Holdout did not show a flip-slice gain without losing overall macro-F1. The layer is not in the demo."
        ),
        "fit_rows": int(len(fit_idx)),
        "holdout_rows": int(len(hold_idx)),
        "forward_window": FORWARD,
        "backward_window": BACKWARD,
        "holdout_flip_macro_f1": None if flip_all is None else flip_all["macro_f1"],
        "holdout_system_macro_f1": None if system_all is None else system_all["macro_f1"],
        "holdout_negation_gain": neg_gain,
        "holdout_contrast_gain": contrast_gain,
        "holdout_echo_gain": echo_gain,
        "holdout_scope_conflict_gain": conflict_gain,
        "scope_words_added": added,
        "sarcasm_label": "none. SentiMix does not annotate sarcasm. The praise cue is a feature, not a class.",
        "test_protocol": "scored once after the holdout decision",
        "system_holdout_note": "The current system stacker was fit on all validation rows, so its holdout score is optimistic.",
    }
    if not kept:
        for name in ("flip_router.joblib", "flip_scaler.joblib"):
            stale = os.path.join(SYSTEM, name)
            if os.path.exists(stale):
                os.remove(stale)
    with open(os.path.join(SYSTEM, "flip_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
    pd.DataFrame([decision]).to_csv(os.path.join(RESULTS, "flip_decision.csv"), index=False)

    print("[Flip] Census")
    print(census.to_string(index=False))
    print("[Flip] Holdout and test")
    show = metrics[metrics["slice"].isin(["all", "negation", "contrast", "scope_conflict", "praise_no_flip"])]
    print(show.to_string(index=False))
    print(f"[Flip] kept={kept} negation_gain={neg_gain} contrast_gain={contrast_gain} echo_gain={echo_gain}")
    print("[Flip] words added for scope:", added)
    user_line = analyze("me acha nahi hu na", weights)
    print("[Flip] example:", user_line["kind"], user_line["trigger"], round(user_line["scoped"], 3))
    return decision


if __name__ == "__main__":
    fit_and_score()
