"""Zeyrek 0.1.3 access with three fixes applied.

1. History dependence: the analyzer mutates shared phonetic-attribute sets in place (the cached result of
   ``calculate_phonetic_attributes`` and the root's ``stem_transition.attrs``), so the analysis of a word depended on
   what had been analyzed before. Each analysis now works on its own copy.
2. Unknown words come back as a pseudo-analysis with lemma ``Unk``; callers treat them as out-of-vocabulary.
3. Candidate order is not stable; ``kdho.core`` picks a candidate with a deterministic rule.
Zeyrek also logs every analysis path at WARNING level ("APPENDING RESULT ..."); that logger is raised to ERROR.
"""
import logging

import zeyrek
import zeyrek.morphotactics as _zm
import zeyrek.rulebasedanalyzer as _zr

_orig_calc = _zr.calculate_phonetic_attributes
_zr.calculate_phonetic_attributes = lambda word, predecessor_attrs=None: set(_orig_calc(word, predecessor_attrs))
_orig_initial = _zm.SearchPath.initial.__func__


def _initial_copy(cls, stem_transition, tail):
    path = _orig_initial(cls, stem_transition, tail)
    path.phonetic_attributes = set(path.phonetic_attributes)
    return path


_zm.SearchPath.initial = classmethod(_initial_copy)
logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)
_ANALYZER = None


def _ensure_punkt():
    import nltk  # Zeyrek tokenizes its input with NLTK's punkt models
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        _ensure_punkt()
        a = zeyrek.MorphAnalyzer()
        try:
            a.analyze("ev")  # smoke test: fails if NLTK punkt data is unavailable
        except LookupError as e:
            raise RuntimeError(
                "kdho could not load the Zeyrek morphological analyzer because NLTK's "
                "'punkt'/'punkt_tab' tokenizer data is missing and could not be downloaded. "
                "Install it with:  python -m nltk.downloader punkt punkt_tab  "
                "(behind a proxy, set NLTK_ALLOW_PROXIED_URLOPEN=1). Without it every word "
                "would be treated as unanalyzable and KDHO would silently equal WER."
            ) from e
        _ANALYZER = a
    return _ANALYZER


def candidates(word):
    """All real Zeyrek analyses of one (lowercased) word; empty list if unknown."""
    if not any(ch.isalpha() for ch in word):
        return []
    a = analyzer()  # setup errors must surface, not be swallowed below
    try:
        res = a.analyze(word)
    except Exception:
        return []
    return [c for c in (res[0] if res and res[0] else []) if c.lemma != "Unk"]
