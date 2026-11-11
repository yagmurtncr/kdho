"""Turkish text normalization for ASR scoring and a rule-based Turkish number verbalizer."""
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']|(?<!\w)'|'(?!\w)")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless i (I→ı, İ→i); also folds a decomposed i + U+0307 to i."""
    return text.replace("İ", "i").replace("I", "ı").lower().replace("i̇", "i")


def normalize_text(text: str) -> str:
    """NFC → Turkish lowercase → punctuation removed (word-internal apostrophes kept) → whitespace collapsed."""
    text = turkish_lower(unicodedata.normalize("NFC", text))
    return " ".join(_PUNCT.sub(" ", text).split())


_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_SCALES = [(10 ** 12, "trilyon"), (10 ** 9, "milyar"), (10 ** 6, "milyon"), (1000, "bin")]


def _below_1000(n):
    h, r = divmod(n, 100)
    t, o = divmod(r, 10)
    out = (["yüz"] if h == 1 else [_ONES[h], "yüz"]) if h else []
    return out + ([_TENS[t]] if t else []) + ([_ONES[o]] if o else [])


def number_to_words(n: int) -> str:
    """Turkish reading of a non-negative integer (TDK spelling: words separated, no "bir bin")."""
    if n == 0:
        return "sıfır"
    out = []
    for val, name in _SCALES:
        q, n = divmod(n, val)
        if q:
            out += [name] if (q == 1 and name == "bin") else number_to_words(q).split() + [name]
    return " ".join(out + _below_1000(n))


def _digits(s):
    s = s.lstrip("0") or "0"
    return number_to_words(int(s)) if len(s) <= 15 else " ".join(_ONES[int(c)] or "sıfır" for c in s)


def _frac(s):
    z = len(s) - len(s.lstrip("0"))
    return " ".join(["sıfır"] * z + ([_digits(s.lstrip("0"))] if s.lstrip("0") else []))


def verbalize_numbers(text: str) -> str:
    """Spell out digits in Turkish.

    ``%50`` → yüzde elli · ``1.000.000`` → bir milyon · ``3,5`` → üç virgül beş · ``14:30`` → on dört otuz ·
    ``1990'da`` → bin dokuz yüz doksanda (the apostrophe suffix already follows the pronunciation).
    Ordinals ("1.") are left alone because they are ambiguous with a sentence-final period.
    """
    text = re.sub(r"%\s?(\d+(?:[.,]\d+)?)", lambda m: "yüzde " + m.group(1), text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b(?![.,]\d)", lambda m: m.group(0).replace(".", ""), text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _digits(m.group(1)) + " " + _digits(m.group(2)), text)
    text = re.sub(r"\b(\d+),(\d+)\b", lambda m: _digits(m.group(1)) + " virgül " + _frac(m.group(2)), text)
    text = re.sub(r"(\d+)'(\w+)", lambda m: _digits(m.group(1)) + m.group(2), text)
    return " ".join(re.sub(r"\d+", lambda m: " " + _digits(m.group(0)) + " ", text).split())
