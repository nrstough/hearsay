"""Detector contract (run spec docs/specs/2026-09-25_software-only-rescope.md, B1-B9)."""

from __future__ import annotations

import math
import subprocess

import numpy as np
import pytest

from hearsay import SR
from hearsay import audio as audio_mod
from hearsay.detectors import base
from hearsay.detectors.base import ClipContext, DetectorResult, Registry, safe_run


def ok(**kw):
    args = {"name": "d", "score": 0.5, "evidence": "because"} | kw
    return DetectorResult(**args)


class Toy:
    """Configurable detector for contract tests."""

    def __init__(self, name="toy", applies=True, run=None):
        self.name = name
        self._applies = applies
        self._run = run or (lambda ctx: DetectorResult(self.name, 0.7, "toy evidence"))
        self.run_calls = 0

    def applies(self, ctx):
        if isinstance(self._applies, Exception):
            raise self._applies
        return self._applies

    def run(self, ctx):
        self.run_calls += 1
        return self._run(ctx)


class EnergyTail:
    """Direction fixture: share of spectral energy above 4 kHz (broadband 'synthetic-marked'
    noise scores above a 200 Hz 'real-marked' tone). Pins the convention, not real audio."""

    name = "energy_tail"

    def applies(self, ctx):
        return True

    def run(self, ctx):
        spec = np.abs(np.fft.rfft(ctx.audio)) ** 2
        freqs = np.fft.rfftfreq(ctx.audio.size, 1 / SR)
        share = float(spec[freqs > 4000].sum() / spec.sum())
        return DetectorResult(self.name, share, f"{share:.0%} of energy above 4 kHz",
                              {"hf_share": share})  # fmt: skip


@pytest.fixture(autouse=True)
def _restore_global_registry():
    saved = dict(base.REGISTRY._by_name)
    yield
    base.REGISTRY._by_name.clear()
    base.REGISTRY._by_name.update(saved)


@pytest.fixture
def ctx():
    return ClipContext.from_array(np.zeros(SR, dtype=np.float32))


# --- B1 score ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -0.01, 1.01, np.float32("nan"), True, np.bool_(True), "0.5"]
)
def test_b1_invalid_scores_rejected_not_clipped(bad):
    with pytest.raises(ValueError):
        ok(score=bad)


@pytest.mark.parametrize("good", [0.0, 1.0, np.float32(0.25), 1])
def test_b1_valid_scores_coerced_to_python_float(good):
    r = ok(score=good)
    assert type(r.score) is float and r.score == float(good)


# --- B2 direction --------------------------------------------------------------------------


def test_b2_score_increases_with_synthetic_marked_input():
    rng = np.random.default_rng(0)
    synthetic_marked = ClipContext.from_array(rng.standard_normal(SR).astype(np.float32) * 0.1)
    t = np.arange(SR) / SR
    real_marked = ClipContext.from_array((0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
    det = EnergyTail()
    assert safe_run(det, synthetic_marked).score > safe_run(det, real_marked).score
    assert "increases with synthetic likelihood" in base.__doc__


# --- B3 errors -----------------------------------------------------------------------------


def test_b3_exception_becomes_error_result_keeping_the_row(ctx):
    def boom(c):
        raise RuntimeError("boom")

    det = Toy(run=boom)
    r = safe_run(det, ctx)
    assert (r.status, r.score, r.features) == ("error", 0.5, {})
    assert r.error == "RuntimeError: boom" and r.name == "toy" and r.evidence.strip()
    with pytest.raises(RuntimeError):  # counterfactual: without safe_run the row is lost
        det.run(ctx)


def test_b3_empty_exception_message_still_valid(ctx):
    def blank(c):
        raise ValueError

    r = safe_run(Toy(run=blank), ctx)
    assert r.status == "error" and r.error.startswith("ValueError")


@pytest.mark.parametrize(
    "ret",
    [
        lambda c: None,
        lambda c: {"score": 0.9},
        lambda c: DetectorResult("other_name", 0.9, "wrong name"),
        lambda c: DetectorResult("toy", 0.9, "smuggled", status="error", error="x"),
        lambda c: DetectorResult("toy", 0.9, "smuggled", status="skipped"),
    ],
)
def test_b3_bad_return_values_become_errors(ctx, ret):
    r = safe_run(Toy(run=ret), ctx)
    assert r.status == "error" and r.score == 0.5


def test_b3_applies_raising_is_an_error(ctx):
    r = safe_run(Toy(applies=RuntimeError("probe died")), ctx)
    assert r.status == "error" and "probe died" in r.error


def test_b3_keyboard_interrupt_propagates(ctx):
    def interrupt(c):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        safe_run(Toy(run=interrupt), ctx)


# --- B4 evidence ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_b4_evidence_required(bad):
    with pytest.raises(ValueError):
        ok(evidence=bad)


# --- B5 features ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"a": {"nested": 1.0}},
        {"a": "1.0"},
        {"a": True},
        {"a": float("nan")},
        {"": 1.0},
        {3: 1.0},
        [("a", 1.0)],
    ],
)
def test_b5_bad_features_rejected(bad):
    with pytest.raises(ValueError):
        ok(features=bad)


def test_b5_int_and_numpy_features_coerced_and_isolated_from_caller():
    src = {"n": 3, "f": np.float64(0.5)}
    r = ok(features=src)
    assert r.features == {"n": 3.0, "f": 0.5}
    assert all(type(v) is float for v in r.features.values())
    src["n"] = 99
    assert r.features["n"] == 3.0


# --- B6 registry ---------------------------------------------------------------------------


def test_b6_registry_rejects_duplicates_and_sorts():
    reg = Registry()
    reg.register(Toy("zeta"))
    reg.register(Toy("alpha"))
    with pytest.raises(ValueError):
        reg.register(Toy("zeta"))
    with pytest.raises(KeyError):
        reg.get("missing")
    assert reg.names() == ["alpha", "zeta"]
    assert [d.name for d in reg.all_detectors()] == ["alpha", "zeta"]


def test_b6_global_registry_is_restored_between_tests():
    base.register(Toy("leaky"))
    assert "leaky" in base.REGISTRY.names()


def test_b6_global_registry_did_not_leak():
    assert "leaky" not in base.REGISTRY.names()


# --- B7 determinism and single decode --------------------------------------------------------


def test_b7_same_clip_scored_twice_is_identical():
    rng = np.random.default_rng(1)
    c = ClipContext.from_array(rng.standard_normal(SR).astype(np.float32))
    assert safe_run(EnergyTail(), c) == safe_run(EnergyTail(), c)


def test_b7_audio_is_read_only(ctx):
    with pytest.raises(ValueError):
        ctx.audio[0] = 1.0


def test_b7_decodes_once(tmp_path, monkeypatch):
    p = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate={SR}:duration=1", str(p)], check=True)  # fmt: skip
    calls = []
    real = audio_mod.load_audio
    monkeypatch.setattr(audio_mod, "load_audio", lambda path: calls.append(path) or real(path))
    c = ClipContext(p)
    a1, a2 = c.audio, c.audio
    assert a1 is a2 and len(calls) == 1 and not a1.flags.writeable


def test_b7_decode_failure_cached(tmp_path, monkeypatch):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    calls = []
    real = audio_mod.load_audio
    monkeypatch.setattr(audio_mod, "load_audio", lambda path: calls.append(path) or real(path))
    c = ClipContext(p)
    for _ in range(2):
        with pytest.raises(audio_mod.DecodeError):
            _ = c.audio
    assert len(calls) == 1


def test_b7_memo_computes_once(ctx):
    n = []
    assert ctx.memo("k", lambda: n.append(1) or 42) == 42
    assert ctx.memo("k", lambda: n.append(1) or 43) == 42
    assert n == [1]


# --- B8 skipped ----------------------------------------------------------------------------


def test_b8_not_applicable_is_skipped_without_running(ctx):
    det = Toy(applies=False, run=lambda c: (_ for _ in ()).throw(AssertionError("ran")))
    r = safe_run(det, ctx)
    assert (r.status, r.score, r.features, r.error) == ("skipped", 0.5, {}, None)
    assert det.run_calls == 0


# --- B9 robustness -------------------------------------------------------------------------


class RaisingName:
    @property
    def name(self):
        raise RuntimeError("no name")

    def applies(self, ctx):
        return True

    def run(self, ctx):
        return DetectorResult("x", 0.5, "never")


@pytest.mark.parametrize("det", [RaisingName(), Toy(name=None), Toy(name="")])
def test_b9_bad_detector_names_never_crash_safe_run(ctx, det):
    r = safe_run(det, ctx)
    assert r.status == "error" and r.name.startswith("<")


@pytest.mark.parametrize("bad", [None, "", "   ", 7])
def test_b9_registry_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        Registry().register(Toy(name=bad))


def test_b9_results_unhashable_but_comparable():
    with pytest.raises(TypeError):
        hash(ok())
    assert ok() == ok()


def test_b9_post_construction_mutation_caught_by_safe_run(ctx):
    def mutate(c):
        r = DetectorResult("toy", 0.6, "fine at first", {"x": 1.0})
        r.features["x"] = math.nan
        return r

    assert safe_run(Toy(run=mutate), ctx).status == "error"


@pytest.mark.parametrize(
    ("status", "error"), [("ok", "x"), ("error", None), ("error", ""), ("skipped", "x"), ("bogus", None)]
)
def test_b9_status_error_coherence(status, error):
    with pytest.raises(ValueError):
        ok(status=status, error=error)
