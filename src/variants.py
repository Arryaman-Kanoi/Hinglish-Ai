"""Roman-spelling variants learned from the SentiMix training tweets.

HingBERT and a normal word model treat nhi and nahi as unrelated words.
This map groups spellings that share a consonant skeleton and are within a
small edit distance, then rewrites each alias to the most common spelling.
The map is built from the training split only.
"""

import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9']+")
VOWEL_FIXES = (
    ("aa", "a"),
    ("ee", "i"),
    ("ii", "i"),
    ("oo", "u"),
    ("uu", "u"),
    ("ph", "f"),
    ("w", "v"),
)


def _skeleton(word):
    folded = re.sub(r"(.)\1+", r"\1", word.lower())
    for source, target in VOWEL_FIXES:
        folded = folded.replace(source, target)
    consonants = re.sub(r"[aeiou]", "", folded)
    return consonants if len(consonants) >= 2 else ""


def _edit_distance(left, right):
    if abs(len(left) - len(right)) > 2:
        return 3
    previous = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, start=1):
        current = [i]
        for j, ch_right in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ch_left != ch_right)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def build_variant_map(texts, min_count=5, max_distance=2):
    """Map a rarer spelling to the most frequent spelling in its group."""
    counts = Counter()
    for text in texts:
        counts.update(TOKEN_RE.findall(str(text).lower()))

    groups = defaultdict(list)
    for word, count in counts.items():
        if len(word) < 3 or count < 2:
            continue
        key = _skeleton(word)
        if not key:
            continue
        groups[key].append(word)

    mapping = {}
    rows = []
    for key, words in groups.items():
        eligible = [word for word in words if counts[word] >= 2]
        if len(eligible) < 2:
            continue
        canonical = max(eligible, key=lambda word: (counts[word], -len(word), word))
        if counts[canonical] < min_count:
            continue
        aliases = []
        for word in eligible:
            if word == canonical:
                continue
            if _edit_distance(word, canonical) > max_distance:
                continue
            mapping[word] = canonical
            aliases.append(word)
        if aliases:
            rows.append({
                "skeleton": key,
                "canonical": canonical,
                "canonical_count": int(counts[canonical]),
                "aliases": " ".join(sorted(aliases, key=lambda word: -counts[word])),
                "alias_counts": " ".join(str(counts[word]) for word in sorted(aliases, key=lambda word: -counts[word])),
            })
    rows.sort(key=lambda row: row["canonical_count"], reverse=True)
    return mapping, rows


def apply_variants(text, mapping):
    def replace(match):
        word = match.group(0)
        mapped = mapping.get(word.lower())
        if not mapped:
            return word
        return mapped

    return TOKEN_RE.sub(replace, str(text).lower())
