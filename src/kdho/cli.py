"""Command-line interface.

    kdho explain "ali eve geldi" "ali eve gelmedi"
    kdho score --ref refs.txt --hyp hyps.txt               # one sentence per line
    kdho score --jsonl outputs.jsonl --ref-key ref --hyp-key hyp --numbers --details
"""
import argparse
import json
import sys

from . import __version__
from .core import LAMBDA_DEFAULT, explain, score


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kdho", description="KDHO: root- and affix-aware error rate for Turkish ASR "
                                 "(error rates, lower is better).")
    ap.add_argument("--version", action="version", version=f"kdho {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lam", type=float, default=LAMBDA_DEFAULT, help="partial affix cost scale (default 0.5)")
    common.add_argument("--numbers", action="store_true", help="spell out digits in Turkish on both sides")
    common.add_argument("--no-normalize", action="store_true", help="do not lowercase / strip punctuation")

    e = sub.add_parser("explain", parents=[common], help="word-by-word alignment of one pair")
    e.add_argument("reference")
    e.add_argument("hypothesis")

    s = sub.add_parser("score", parents=[common], help="corpus-level KDHO, WER and CER")
    s.add_argument("--ref", help="reference file, one sentence per line")
    s.add_argument("--hyp", help="hypothesis file, one sentence per line")
    s.add_argument("--jsonl", help="JSONL file with one pair per line")
    s.add_argument("--ref-key", default="reference")
    s.add_argument("--hyp-key", default="hypothesis")
    s.add_argument("--details", action="store_true", help="also print operation counts by type")
    a = ap.parse_args(argv)

    if a.cmd == "explain":
        al = explain(a.reference, a.hypothesis, lam=a.lam, normalize=not a.no_normalize, verbalize_numbers=a.numbers)
        w = max([len(r or "") for _, r, _, _ in al.steps] + [9])
        print(f"{'reference':<{w}}  {'hypothesis':<{w}}  {'type':<9}  cost")
        for t, r, h, c in al.steps:
            print(f"{r or '—':<{w}}  {h or '—':<{w}}  {t:<9}  {c:.3f}")
        print(f"\nKDHO = {al.kdho:.4f}  ({al.cost:.3f} / {al.n_ref_words} reference words)")
        return 0

    if a.jsonl:
        rows = [json.loads(line) for line in open(a.jsonl, encoding="utf-8") if line.strip()]
        refs, hyps = [r[a.ref_key] for r in rows], [r[a.hyp_key] for r in rows]
    elif a.ref and a.hyp:
        refs, hyps = _lines(a.ref), _lines(a.hyp)
    else:
        ap.error("score needs --ref and --hyp, or --jsonl")
    out = score(refs, hyps, lam=a.lam, normalize=not a.no_normalize, verbalize_numbers=a.numbers, return_details=a.details)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
