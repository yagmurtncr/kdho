"""KDHO — Kök ve Ek Duyarlı Hata Oranı: a root- and affix-aware word error rate for Turkish ASR.

>>> import kdho
>>> kdho.kdho("ali eve geldi", "ali eve gelmedi")        # negation flip → full error
0.3333333333333333
>>> kdho.kdho("evlerde geldik", "evlerden geldik")       # one case suffix wrong → partial error
0.125

Corpus level: ``kdho.score(refs, hyps, verbalize_numbers=True)`` → {'kdho', 'wer', 'cer', ...}
"""
from .core import LAMBDA_DEFAULT, Alignment, align, analyze, explain, kdho, score, substitution_cost
from .normalize import normalize_text, number_to_words, turkish_lower, verbalize_numbers

__version__ = "0.1.2"
__all__ = ["kdho", "score", "explain", "align", "analyze", "substitution_cost", "Alignment", "LAMBDA_DEFAULT",
           "normalize_text", "turkish_lower", "verbalize_numbers", "number_to_words", "__version__"]
