# kdho

[![tests](https://github.com/yagmurtncr/kdho/actions/workflows/tests.yml/badge.svg)](https://github.com/yagmurtncr/kdho/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/kdho)](https://pypi.org/project/kdho/)
[![Hugging Face demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-yagmurtuncer%2Fkdho-yellow)](https://huggingface.co/spaces/yagmurtuncer/kdho)

**Links:** [PyPI](https://pypi.org/project/kdho/) · [Try it in the browser (Hugging Face)](https://huggingface.co/spaces/yagmurtuncer/kdho) ·
[Hugging Face `evaluate`](https://huggingface.co/spaces/yagmurtuncer/kdho): `evaluate.load("yagmurtuncer/kdho")` ·
[Turkish ASR Leaderboard](https://huggingface.co/spaces/yagmurtuncer/turkish-asr-leaderboard) (KDHO column for 26 models)

**KDHO** (*Kök ve Ek Duyarlı Hata Oranı*, Root- and Affix-Aware Error Rate) is a word error rate for Turkish speech
recognition that understands morphology.

Turkish is agglutinative: one word carries a root and a chain of suffixes. WER counts *evlerde → evlerden* (one case
suffix wrong) exactly like *evlerde → okullarda* (a different word) or *geldi → gelmedi* (meaning reversed). KDHO keeps
the word as the unit, but prices each substitution with a morphological analysis of the two words:

| Case | Cost |
|---|---|
| identical words | 0 |
| different roots | 1 |
| negation differs (*-mA*, *-(y)AmA*, *değil*) | 1 |
| either word cannot be analyzed | 1 |
| same root, affix difference | λ · d_M (normalized edit distance between affix tags, λ = 0.5) |

Deletions and insertions cost 1 and the cost is used inside the alignment, so **0 ≤ KDHO ≤ WER** always holds.
All scores are error rates: **lower is better**.

## Install

```bash
pip install kdho
```

## Use

```python
import kdho

kdho.kdho("ali eve geldi", "ali eve gelmedi")     # 0.333  negation flip → full error
kdho.kdho("evlerde geldik", "evlerden geldik")    # 0.125  one case suffix wrong → partial error

kdho.score(references, hypotheses)                # corpus level
# {'kdho': 0.113, 'wer': 0.128, 'cer': 0.070, 'n_sentences': 5200, 'n_ref_words': 61234}

kdho.score(references, hypotheses, verbalize_numbers=True, return_details=True)
# digits spelled out on both sides (48 → kırk sekiz); 'ops' counts equal/affix/root/negation/oov/deletion/insertion

for step in kdho.explain("bu kitapları yarın okumayacağım", "bu kitabı yarın okuyacağım").steps:
    print(step)
# ('equal', 'bu', 'bu', 0.0)
# ('affix', 'kitapları', 'kitabı', 0.25)
# ('equal', 'yarın', 'yarın', 0.0)
# ('negation', 'okumayacağım', 'okuyacağım', 1.0)
```

### Command line

```bash
kdho explain "ali eve geldi" "ali eve gelmedi"
kdho score --ref refs.txt --hyp hyps.txt --numbers --details         # one sentence per line
kdho score --jsonl outputs.jsonl --ref-key ref_norm --hyp-key hyp_norm
```

### Options

| Argument | Default | Meaning |
|---|---|---|
| `lam` | 0.5 | scale of the partial affix cost (0 = ignore affix errors, 1 = proportional full cost) |
| `normalize` | True | NFC, Turkish lowercasing (I→ı, İ→i), punctuation removal; word-internal apostrophes (*Ankara'ya*) kept |
| `verbalize_numbers` | False | spell out digits in Turkish: *%50* → *yüzde elli*, *1990'da* → *bin dokuz yüz doksanda* |
| `return_details` | False | add operation counts by type |

Helpers: `kdho.normalize_text`, `kdho.turkish_lower`, `kdho.verbalize_numbers`, `kdho.number_to_words`,
`kdho.substitution_cost`, `kdho.analyze`.

## Validation

On a balanced Turkish test set (7 sources × 743 utterances, 14 open ASR models):

- Agreement with human meaning-change ratings (Spearman ρ, 48 pairs): KDHO 0.83, WER 0.75.
- Negation errors, 3 independent raters: agreement with the human ordering KDHO 78 %, CER 52 %.
- 72,800 system outputs: a negation error is ranked above a character-matched affix error 97 % of the time by KDHO,
  67 % by CER.

## Notes and limitations

- Morphological analysis uses [Zeyrek](https://github.com/obulat/zeyrek) 0.1.3 (pinned). `kdho` fixes three issues of
  that version at import time: history-dependent analyses, a fake `Unk` analysis for unknown words, and unstable
  candidate order. Results are deterministic.
- Analysis is context-free; spelling variants (*olacak/olucak*) and synonyms count as full errors.
- The first call downloads NLTK's `punkt` tokenizer if it is missing. If that fails (offline, proxy), `kdho` raises a
  clear error instead of silently scoring every word as unanalyzable. Install it manually with
  `python -m nltk.downloader punkt punkt_tab`.

## Links

- [Turkish ASR Leaderboard](https://huggingface.co/spaces/yagmurtuncer/turkish-asr-leaderboard): 19 open models ranked with WER / CER
- [TurkASR-Bench](https://huggingface.co/datasets/yagmurtuncer/turkasr-bench): the balanced test set used for validation

## Citation

```bibtex
@misc{kdho2026,
  title = {KDHO: A Root- and Affix-Aware Error Rate for Turkish Speech Recognition},
  year  = {2026},
  note  = {Under review}
}
```

License: Apache-2.0.
