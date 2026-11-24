"""
src/preprocess.py
-----------------
Lightweight, sentiment-preserving text preprocessing for Roman-script Hinglish tweets.

Design Principles:
1. Preserve Emojis: Emojis carry vital emotional signal in social media sentiment.
2. Preserve Casing: All-caps (e.g. "BEKAAR", "SUPERB") signifies intense sentiment.
3. Normalize Repetitions: "sooooo goooood" -> "soo good" (caps at 2 consecutive chars).
4. Anonymize URLs and user mentions as <URL> and <USER>.
5. Avoid Stopword Removal: Modern pretrained transformers leverage full syntax & context.
"""

import re
import html

# Try importing emoji library; if not available, fallback to regex-based emoji stripping
try:
    import emoji
    HAS_EMOJI_PKG = True
except ImportError:
    HAS_EMOJI_PKG = False

# Regex patterns. SentiMix CoNLL splits URLs ("https // t . co / abc"), so the
# URL pattern also accepts those broken forms.
URL_PATTERN = re.compile(
    r'(?:https?\s*:?\s*/+\s*\S+(?:\s+[A-Za-z0-9._/-]+){0,6}|www\.\S+)',
    re.IGNORECASE,
)
USER_PATTERN = re.compile(r'@\s*[A-Za-z0-9_]+')
REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{2,}')
WHITESPACE_PATTERN = re.compile(r'\s+')
HASHTAG_PATTERN = re.compile(r'#\s*(\w+)')


def normalize_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """
    Collapses sequences of 3+ identical consecutive characters down to max_repeat.
    Example:
        'sooo goooood' -> 'soo good'
        'waaahhhh' -> 'waahh'
    Preserves emphasis while preventing out-of-vocabulary token explosion.
    """
    return REPEATED_CHARS_PATTERN.sub(r'\1' * max_repeat, text)


def remove_emojis(text: str) -> str:
    """
    Strips emojis from text. Used specifically for Ablation Study (Emoji vs No-Emoji).
    """
    if HAS_EMOJI_PKG:
        return emoji.replace_emoji(text, replace='')
    # Fallback Unicode range for common emojis
    emoji_regex = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_regex.sub(r'', text)


def clean_tweet(
    text: str,
    normalize_urls: bool = True,
    normalize_mentions: bool = True,
    normalize_repeats: bool = True,
    strip_emojis: bool = False,
    preserve_casing: bool = True
) -> str:
    """
    Applies light, sentiment-preserving preprocessing tailored for Hinglish tweets.

    Parameters:
    -----------
    text : str
        Input raw Hinglish tweet text.
    normalize_urls : bool
        If True, replaces URLs with the token '<URL>'.
    normalize_mentions : bool
        If True, replaces @mentions with the token '<USER>'.
    normalize_repeats : bool
        If True, condenses 3+ repeated characters to 2 (e.g. 'bohottt' -> 'bohatt').
    strip_emojis : bool
        If True, strips emojis (used for ablation). Default False (emojis kept).
    preserve_casing : bool
        If True, keeps case sensitivity. Default True.

    Returns:
    --------
    str: Cleaned text ready for tokenization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # 2. Normalize URLs
    if normalize_urls:
        cleaned = URL_PATTERN.sub('<URL>', cleaned)

    # 3. Normalize User Mentions
    if normalize_mentions:
        cleaned = USER_PATTERN.sub('<USER>', cleaned)

    # 4. Handle Hashtags: Keep hashtag text without '#' symbol
    # e.g., #SuperbMovie -> SuperbMovie
    cleaned = HASHTAG_PATTERN.sub(r'\1', cleaned)

    # 5. Condense character repetitions
    if normalize_repeats:
        cleaned = normalize_repeated_chars(cleaned, max_repeat=2)

    # 6. Emoji handling (kept by default, stripped only for ablation)
    if strip_emojis:
        cleaned = remove_emojis(cleaned)

    # 7. Normalize whitespace
    cleaned = WHITESPACE_PATTERN.sub(' ', cleaned).strip()

    # 8. Casing
    if not preserve_casing:
        cleaned = cleaned.lower()

    return cleaned


def preprocess_corpus(
    texts,
    strip_emojis: bool = False,
    raw_mode: bool = False
):
    """
    Batch preprocessing helper that supports ablation modes:
    - raw_mode=True: returns unchanged raw text
    - strip_emojis=True: returns preprocessed text without emojis
    """
    if raw_mode:
        return [str(t) if t is not None else "" for t in texts]

    return [clean_tweet(t, strip_emojis=strip_emojis) for t in texts]


if __name__ == "__main__":
    sample = "Bhaaaaiii yeh movie sach me zabardasttt hai!!! 🔥❤️ check out https://t.co/xyz123 @director_sahab #Blockbuster"
    print("Raw:         ", sample)
    print("Cleaned:     ", clean_tweet(sample))
    print("No Emojis:   ", clean_tweet(sample, strip_emojis=True))
    print("Lowercased:  ", clean_tweet(sample, preserve_casing=False))
