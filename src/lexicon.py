"""Polarity lexicon mined from the SentiMix training split, with a negation window.

Words are scored from train tweets only. A negation token flips the next three
word scores. The lexicon is a specialist beside the character TF-IDF model.
"""

import math
import re
from collections import Counter

NEGATION = {
    "nahi", "nahin", "nhi", "nai", "na", "mat", "no", "not", "never",
}
SKIP = {
    "url", "user", "rt", "amp", "http", "https", "co",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "it",
    "this", "that", "with", "at", "from", "be", "are", "was", "were", "you",
    "hai", "hain", "ho", "tha", "thi", "the", "ka", "ke", "ki", "ko", "se",
    "me", "mein", "main", "bhi", "to", "toh", "aur", "ye", "yeh", "woh", "vo",
    "hi", "par", "pe", "kya", "kaise", "raha", "rahi", "rahe", "kar", "karke",
    "wala", "wali", "ek", "bhi", "jo", "ki", "ke", "ka",
}
TOKEN_RE = re.compile(r"[a-z0-9']+")
ELONGATION_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)


def tokens_of(text):
    return TOKEN_RE.findall(str(text).lower())


def has_negation(text):
    return any(tok in NEGATION for tok in tokens_of(text))


def has_elongation(text):
    return ELONGATION_RE.search(str(text)) is not None


def build_lexicon(texts, labels, min_count=8, min_abs=0.35, limit=2500):
    """Return {word: polarity} where polarity is PMI(positive) - PMI(negative)."""
    by_class = {0: Counter(), 1: Counter(), 2: Counter()}
    class_tokens = {0: 0, 1: 0, 2: 0}
    for text, label in zip(texts, labels):
        label = int(label)
        seen = tokens_of(text)
        class_tokens[label] += len(seen)
        by_class[label].update(seen)

    total_tokens = sum(class_tokens.values()) or 1
    vocabulary = set()
    for counter in by_class.values():
        vocabulary.update(counter)
    vocab_size = max(len(vocabulary), 1)

    scored = []
    for word in vocabulary:
        if word in SKIP or word in NEGATION or len(word) < 2:
            continue
        pos = by_class[2][word]
        neg = by_class[0][word]
        if pos + neg < min_count:
            continue
        total = pos + neg + by_class[1][word]

        def pmi(count, class_n):
            p_word_class = (count + 0.5) / (class_n + 0.5 * vocab_size)
            p_word = (total + 0.5) / (total_tokens + 0.5 * vocab_size)
            return math.log(p_word_class) - math.log(p_word)

        polarity = pmi(pos, class_tokens[2]) - pmi(neg, class_tokens[0])
        if abs(polarity) < min_abs:
            continue
        scored.append((word, float(polarity), int(pos), int(neg)))

    scored.sort(key=lambda row: abs(row[1]), reverse=True)
    scored = scored[:limit]
    weights = {word: polarity for word, polarity, _, _ in scored}
    table = [
        {"word": word, "polarity": polarity, "positive_count": pos, "negative_count": neg}
        for word, polarity, pos, neg in scored
    ]
    return weights, table


def score_text(text, weights, window=3):
    """Sum lexicon weights. Negation flips the next `window` words.

    Returns polarity, positive mass, negative mass, and the strongest token.
    """
    polarity = 0.0
    pos_mass = 0.0
    neg_mass = 0.0
    trigger = ""
    trigger_abs = 0.0
    negation_left = 0
    current_negation = ""
    for tok in tokens_of(text):
        if tok in NEGATION:
            negation_left = window
            current_negation = tok
            continue
        weight = weights.get(tok, 0.0)
        if weight == 0.0:
            if negation_left:
                negation_left -= 1
            continue
        applied = current_negation if negation_left else ""
        signed = -weight if applied else weight
        if negation_left:
            negation_left -= 1
        polarity += signed
        if signed > 0:
            pos_mass += signed
        else:
            neg_mass += -signed
        if abs(signed) > trigger_abs:
            trigger_abs = abs(signed)
            trigger = f"{applied} {tok}".strip()
    return polarity, pos_mass, neg_mass, trigger
