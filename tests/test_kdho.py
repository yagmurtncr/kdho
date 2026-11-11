import json
import random
from pathlib import Path

import jiwer
import pytest

import kdho
from kdho import cli
from kdho.core import _CACHE, analyze

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("g", [g for g in GOLDEN if g["n"]], ids=lambda g: f"{g['ref']}|{g['hyp']}")
def test_matches_reference_implementation(g):
    a = kdho.align(g["ref"], g["hyp"])
    assert a.cost == pytest.approx(g["cost"])
    assert a.n_ref_words == g["n"]


def test_negation_is_full_error_affix_is_partial():
    assert kdho.kdho("ali eve geldi", "ali eve gelmedi") == pytest.approx(1 / 3)
    assert kdho.kdho("evlerde geldik", "evlerden geldik") == pytest.approx(0.125)
    assert kdho.kdho("evlerde oturuyoruz", "okullarda oturuyoruz") == pytest.approx(0.5)


def test_identical_is_zero_and_punctuation_case_ignored():
    assert kdho.kdho("İstanbul'a GİTTİK.", "istanbul'a gittik") == 0.0


@pytest.mark.parametrize("ref, hyp", [(g["ref"], g["hyp"]) for g in GOLDEN if g["ref"]])
def test_never_above_wer(ref, hyp):
    s = kdho.score([ref], [hyp])
    assert s["kdho"] <= s["wer"] + 1e-12


def test_monotone_in_lambda():
    refs = [g["ref"] for g in GOLDEN]
    hyps = [g["hyp"] for g in GOLDEN]
    vals = [kdho.score(refs, hyps, lam=lam)["kdho"] for lam in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals)


def test_analysis_is_history_independent():
    words = ["kitaplarımızdan", "gelmiyorsun", "evlerde", "okuyacağım", "arkadaşlarımla", "kuruyorsun", "yapamaz"]
    first = {w: analyze(w) for w in words}
    rng = random.Random(0)
    for _ in range(3):
        _CACHE.clear()
        for w in rng.sample(["gözlükçüden", "yürümeliydik", "sevindirici", "anlaşılamayan", "düşünceler"] * 20, 60):
            analyze(w)
        for w in rng.sample(words, len(words)):
            assert analyze(w) == first[w]


def test_number_verbalization():
    v = kdho.verbalize_numbers
    assert v("1990'da geldi") == "bin dokuz yüz doksanda geldi"
    assert v("%50 indirim") == "yüzde elli indirim"
    assert v("1.000.000 kişi") == "bir milyon kişi"
    assert v("3,05 kilo") == "üç virgül sıfır beş kilo"
    assert v("saat 14:30") == "saat on dört otuz"
    assert kdho.number_to_words(2024) == "iki bin yirmi dört"
    assert kdho.score(["kırk sekiz saat"], ["48 saat"], verbalize_numbers=True)["wer"] == 0.0


def test_turkish_lowercase():
    assert kdho.turkish_lower("IŞIK İzmir") == "ışık izmir"
    assert kdho.normalize_text("Ankara'ya, (hızlıca) gitti!") == "ankara'ya hızlıca gitti"


def test_score_details_and_empty_reference():
    s = kdho.score(["ali eve geldi", ""], ["ali eve gelmedi", "boş"], return_details=True)
    assert s["n_sentences"] == 1 and s["ops"]["negation"] == 1 and s["wer"] == pytest.approx(jiwer.wer("ali eve geldi", "ali eve gelmedi"))
    with pytest.raises(ValueError):
        kdho.score(["a"], [])


def test_explain_steps():
    a = kdho.explain("bu kitapları yarın okumayacağım", "bu kitabı yarın okuyacağım")
    assert [t for t, *_ in a.steps] == ["equal", "affix", "equal", "negation"]


def test_cli(tmp_path, capsys):
    (tmp_path / "r.txt").write_text("ali eve geldi\nevlerde geldik\n", encoding="utf-8")
    (tmp_path / "h.txt").write_text("ali eve gelmedi\nevlerden geldik\n", encoding="utf-8")
    assert cli.main(["score", "--ref", str(tmp_path / "r.txt"), "--hyp", str(tmp_path / "h.txt"), "--details"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kdho"] == pytest.approx(1.25 / 5)
    assert cli.main(["explain", "ali eve geldi", "ali eve gelmedi"]) == 0
    assert "negation" in capsys.readouterr().out


def test_missing_punkt_raises_instead_of_silent_wer(monkeypatch):
    """If the analyzer cannot load, kdho must fail loudly rather than fall back to WER."""
    from kdho import _zeyrek

    class Broken:
        def analyze(self, word):
            raise LookupError("punkt not found")

    monkeypatch.setattr(_zeyrek, "_ANALYZER", None)
    monkeypatch.setattr(_zeyrek, "_ensure_punkt", lambda: None)
    monkeypatch.setattr(_zeyrek.zeyrek, "MorphAnalyzer", Broken)
    with pytest.raises(RuntimeError, match="punkt"):
        _zeyrek.candidates("evlerde")
