"""KDHO core: word analysis, substitution cost, morphology-aware alignment, corpus score.

    A(w) = (lemma, affix tags M, negation p, analyzable)
    c(r, h) = 0 if r == h;  1 if an analysis is missing, the lemmas differ or the negation differs;
              otherwise λ · Lev(M_r, M_h) / max(|M_r|, |M_h|, 1)
    D(i,0) = i, D(0,j) = j, D(i,j) = min{D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+c(r_i,h_j)},  KDHO = D(N,K) / N
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import jiwer

from . import _zeyrek
from .normalize import normalize_text, turkish_lower, verbalize_numbers

LAMBDA_DEFAULT = 0.5
NEGATION_TAGS = {"Neg", "Unable"}   # -mA negation and -(y)AmA inability
NEGATION_LEMMAS = {"değil"}         # lexical negation
_EDGE_PUNCT = ".,;:!?\"'()[]{}«»“”‘’…-–—"
_CACHE: Dict[str, Tuple[str, tuple, bool, bool]] = {}
OP_TYPES = ("equal", "affix", "root", "negation", "oov", "deletion", "insertion")


def analyze(word: str) -> Tuple[str, tuple, bool, bool]:
    """(lemma, affix tags without Neg, negated, analyzable) for one word, context-free.

    Candidate choice is deterministic: non-proper-noun first, then fewest morphemes, then alphabetical.
    A word counts as negated only if *every* analysis carries Neg/Unable (or its lemma is "değil"), so that the
    -mA negation/infinitive ambiguity does not create spurious negation errors.
    """
    w = turkish_lower(word)
    if w not in _CACHE:
        cands = _zeyrek.candidates(w)
        if not cands:
            _CACHE[w] = (w.strip(_EDGE_PUNCT) or w, (), False, False)
        else:
            p = min(cands, key=lambda c: (",Prop" in c.formatted, len(c.morphemes), c.formatted))
            an = [(c.lemma, set(c.morphemes[1:])) for c in cands]
            neg = any(lemma in NEGATION_LEMMAS for lemma, _ in an) or all(NEGATION_TAGS & tags for _, tags in an)
            _CACHE[w] = (p.lemma, tuple(f for f in p.morphemes[1:] if f != "Neg"), neg, True)
    return _CACHE[w]


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def substitution_cost(ref_word: str, hyp_word: str, lam: float = LAMBDA_DEFAULT) -> Tuple[float, str]:
    """Cost and type (equal / affix / root / negation / oov) of substituting ``hyp_word`` for ``ref_word``."""
    sr = turkish_lower(ref_word).strip(_EDGE_PUNCT) or ref_word
    sh = turkish_lower(hyp_word).strip(_EDGE_PUNCT) or hyp_word
    if sr == sh:
        return 0.0, "equal"
    lr, mr, pr, okr = analyze(ref_word)
    lh, mh, ph, okh = analyze(hyp_word)
    if not (okr and okh):
        return 1.0, "oov"
    if turkish_lower(lr) != turkish_lower(lh):
        return 1.0, "root"
    if pr != ph:
        return 1.0, "negation"
    return lam * _lev(mr, mh) / max(len(mr), len(mh), 1), "affix"


@dataclass
class Alignment:
    """Result for one reference/hypothesis pair."""
    cost: float
    n_ref_words: int
    ops: Dict[str, int]
    steps: List[Tuple[str, Optional[str], Optional[str], float]] = field(default_factory=list)  # (type, ref, hyp, cost)

    @property
    def kdho(self) -> float:
        return self.cost / self.n_ref_words if self.n_ref_words else 0.0


def align(reference: str, hypothesis: str, lam: float = LAMBDA_DEFAULT) -> Alignment:
    """Morphology-aware alignment of two already-normalized strings (split on whitespace)."""
    R, H = reference.split(), hypothesis.split()
    N, K = len(R), len(H)
    ops = dict.fromkeys(OP_TYPES, 0)
    C = [[substitution_cost(r, h, lam) for h in H] for r in R]
    D = [[0.0] * (K + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        D[i][0] = i
    for j in range(1, K + 1):
        D[0][j] = j
    for i in range(1, N + 1):
        for j in range(1, K + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + C[i - 1][j - 1][0])
    steps = []
    i, j = N, K
    while i > 0 or j > 0:  # backtrace; ties prefer the diagonal
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + C[i - 1][j - 1][0])) < 1e-12:
            c, t = C[i - 1][j - 1]
            steps.append((t, R[i - 1], H[j - 1], c))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + 1)) < 1e-12:
            steps.append(("deletion", R[i - 1], None, 1.0))
            i -= 1
        else:
            steps.append(("insertion", None, H[j - 1], 1.0))
            j -= 1
    steps.reverse()
    for t, *_ in steps:
        ops[t] += 1
    return Alignment(cost=D[N][K], n_ref_words=N, ops=ops, steps=steps)


def prepare(text: str, normalize: bool = True, numbers: bool = False) -> str:
    if numbers:
        text = verbalize_numbers(text)
    return normalize_text(text) if normalize else " ".join(text.split())


def score(references: Iterable[str], hypotheses: Iterable[str], lam: float = LAMBDA_DEFAULT, normalize: bool = True,
          verbalize_numbers: bool = False, return_details: bool = False) -> dict:
    """Corpus-level KDHO (total cost / total reference words) plus WER and CER on the same normalized text.

    All three are error rates: lower is better. Pairs with an empty reference are skipped.
    """
    refs, hyps = list(references), list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    cost, n = 0.0, 0
    ops = dict.fromkeys(OP_TYPES, 0)
    R, H = [], []
    for r, h in zip(refs, hyps):
        r, h = prepare(r, normalize, verbalize_numbers), prepare(h, normalize, verbalize_numbers)
        if not r:
            continue
        a = align(r, h, lam)
        cost, n = cost + a.cost, n + a.n_ref_words
        for k, v in a.ops.items():
            ops[k] += v
        R.append(r)
        H.append(h)
    out = {"kdho": cost / n if n else 0.0,
           "wer": jiwer.wer(R, H) if R else 0.0,
           "cer": jiwer.cer(R, H) if R else 0.0,
           "n_sentences": len(R), "n_ref_words": n}
    if return_details:
        out["ops"] = ops
    return out


def kdho(references, hypotheses, **kwargs) -> float:
    """Corpus-level KDHO for lists of strings, or for a single reference/hypothesis pair."""
    if isinstance(references, str):
        references, hypotheses = [references], [hypotheses]
    return score(references, hypotheses, **kwargs)["kdho"]


def explain(reference: str, hypothesis: str, lam: float = LAMBDA_DEFAULT, normalize: bool = True,
            verbalize_numbers: bool = False) -> Alignment:
    """Word-by-word alignment with the type and cost of every operation (for inspection)."""
    return align(prepare(reference, normalize, verbalize_numbers), prepare(hypothesis, normalize, verbalize_numbers), lam)
