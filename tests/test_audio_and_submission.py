"""M0 loader + submission-writer contract: plan.md "Format pitfalls" and "Rules that never move"."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hearsay import SR
from hearsay.audio import DecodeError, load_audio, probe_audio
from hearsay.submission import append_log, score_files, write_submission


def ffmpeg_sine(out: Path, *, sr: int = 44_100, ch: int = 2, dur: float = 1.0, args=()) -> Path:
    src = f"sine=frequency=440:sample_rate={sr}:duration={dur}"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-ac", str(ch)]
    subprocess.run([*cmd, *args, str(out)], check=True)
    return out


@pytest.mark.parametrize(
    ("name", "sr", "ch", "args"),
    [
        ("stereo44k.wav", 44_100, 2, ()),
        ("mono8k.wav", 8_000, 1, ()),
        ("odd22050.wav", 22_050, 1, ()),
        ("cbr.mp3", 44_100, 2, ("-b:a", "128k")),
        ("vbr.mp3", 44_100, 2, ("-q:a", "4")),
        ("aac.m4a", 48_000, 2, ()),
        ("lossless.flac", 48_000, 1, ()),
        ("opus.ogg", 48_000, 1, ("-c:a", "libopus")),
        ("aac.mp4", 48_000, 2, ("-c:a", "aac", "-f", "mp4")),
    ],
)
def test_any_format_to_16k_mono_float32(tmp_path, name, sr, ch, args):
    p = ffmpeg_sine(tmp_path / name, sr=sr, ch=ch, dur=1.0, args=args)
    x = load_audio(p)
    assert x.dtype == np.float32 and x.ndim == 1
    # Lossy encoders pad/trim by up to a few frames; within 100 ms of 1 s.
    assert abs(x.size - SR) < 0.1 * SR
    assert np.isfinite(x).all() and 0.05 < np.abs(x).max() <= 1.01
    info = probe_audio(p)
    assert info["sample_rate"] == sr and info["channels"] == ch


def test_load_is_deterministic(tmp_path):
    p = ffmpeg_sine(tmp_path / "a.mp3", dur=2.0)
    assert np.array_equal(load_audio(p), load_audio(p))


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(DecodeError):
        load_audio(p)
    assert "error" in probe_audio(p)


def test_garbage_file_raises(tmp_path):
    p = tmp_path / "junk.mp3"
    p.write_bytes(b"not audio at all" * 10)
    with pytest.raises(DecodeError):
        load_audio(p)


def test_truncated_wav_decodes_what_it_can(tmp_path):
    p = ffmpeg_sine(tmp_path / "full.wav", sr=SR, ch=1, dur=1.0)
    t = tmp_path / "trunc.wav"
    t.write_bytes(p.read_bytes()[: 44 + 2 * SR // 2])  # header + half the samples
    x = load_audio(t)
    assert 0.4 * SR < x.size < 0.6 * SR and np.isfinite(x).all()


def test_nan_float_wav_is_sanitized(tmp_path):
    y = np.full(SR, 0.1, dtype=np.float32)
    y[100:200] = np.nan
    y[300] = np.inf
    p = tmp_path / "nan.wav"
    sf.write(p, y, SR, subtype="FLOAT")
    x = load_audio(p)
    assert np.isfinite(x).all() and x[150] == 0.0


def test_silence_loads_as_zeros(tmp_path):
    p = tmp_path / "silence.wav"
    sf.write(p, np.zeros(SR, dtype=np.float32), SR)
    x = load_audio(p)
    assert x.size == SR and not x.any()


def test_no_loudness_normalization(tmp_path):
    p = tmp_path / "quiet.wav"
    sf.write(p, np.full(SR, 0.001, dtype=np.float32), SR, subtype="FLOAT")
    assert np.allclose(load_audio(p), 0.001, atol=1e-6)


# --- submission ---------------------------------------------------------------------------


def test_score_files_one_finite_score_per_file_in_order(tmp_path):
    good = ffmpeg_sine(tmp_path / "good.wav", sr=SR, ch=1)
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    paths = [good, empty, good]
    calls = iter([0.2, float("nan")])
    res = score_files(paths, lambda x: next(calls), fallback=0.5)
    assert res.scores == [0.2, 0.5, 0.5]
    assert res.flags == ["", "decode_error", "nonfinite_score"]
    assert res.n_flagged == 2


def test_score_files_survives_scorer_exception(tmp_path):
    good = ffmpeg_sine(tmp_path / "good.wav", sr=SR, ch=1)

    def boom(x):
        raise RuntimeError("model died")

    res = score_files([good], boom, fallback=0.3)
    assert res.scores == [0.3] and res.flags == ["score_error:RuntimeError"]


def test_write_submission_nsa_tsv_format(tmp_path):
    ids = ["c.wav", "a.wav", "b.wav"]
    out = write_submission(ids, [0.9, 0.1, 0.5], tmp_path / "s.tsv")
    assert out.read_text() == "filename\tcm-score\nc.wav\t0.9\na.wav\t0.1\nb.wav\t0.5\n"


def test_write_submission_never_overwrites(tmp_path):
    write_submission(["a"], [0.5], tmp_path / "s.csv")
    with pytest.raises(FileExistsError):
        write_submission(["a"], [0.6], tmp_path / "s.csv")


@pytest.mark.parametrize(
    ("ids", "scores"),
    [
        (["a", "b"], [0.5]),
        (["a", "a"], [0.5, 0.5]),
        (["a"], [float("nan")]),
        (["a"], [1.5]),
        (["a"], [-0.1]),
        ([], []),
    ],
)
def test_write_submission_rejects_bad_input(tmp_path, ids, scores):
    with pytest.raises(ValueError):
        write_submission(ids, scores, tmp_path / "s.csv")
    assert not (tmp_path / "s.csv").exists()


def test_append_log(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text(
        "timestamp,rung,validation_score,validation_score_clean_only,csv_path,notes\n"
    )
    append_log("M0", tmp_path / "x.csv", validation_score=0.5, notes="const", log_path=log)
    rows = list(csv.DictReader(log.open()))
    assert rows[0]["rung"] == "M0"
    assert rows[0]["validation_score_clean_only"] == "0.5"


@pytest.mark.parametrize(("n", "expect"), [(0, 1), (100, 1), (64_000, 1), (64_001, 2), (96_000, 2), (160_000, 4)])
def test_windows_cover_clip(n, expect):
    from hearsay.audio import windows

    x = np.arange(n, dtype=np.float32)
    w = windows(x, 64_000)
    assert w.shape == (expect, 64_000) and w.dtype == np.float32
    if n >= 64_000:
        assert w[0, 0] == 0 and w[-1, -1] == n - 1


@pytest.mark.needs_weights
@pytest.mark.slow
def test_embed_clip_matches_bulk_extraction_path(tmp_path):
    """The CSV scorer (embed_clip) and bulk extraction must produce the same features."""
    from hearsay.embed import REPO, clip_windows, embed_clip, embed_windows, load_backbone

    if not (REPO / "weights" / "wav2vec2-xls-r-300m" / "config.json").exists():
        pytest.skip("weights not present")
    p = ffmpeg_sine(tmp_path / "a.wav", sr=SR, ch=1, dur=6.0)
    x = load_audio(p)
    model = load_backbone()
    per_clip = embed_clip(model, x)
    bulk = embed_windows(model, list(clip_windows(x)), batch=1).mean(axis=0)
    assert per_clip.shape == (25, 1024)
    assert np.allclose(per_clip, bulk, atol=1e-3)


# --- metrics (NSA rule: false alarm costs 4x a miss) ----------------------------------------


def test_cost_constant_decisions_normalize_to_one():
    from hearsay.metrics import cost_at

    y = np.array([0] * 50 + [1] * 50)
    s = np.zeros(100)
    # always real: cost = pi_synth * 1 = 0.5; default = min(4*0.5, 0.5) = 0.5 -> 1.0
    assert cost_at(y, s, thr=1.0, pi_synth=0.5) == 1.0
    # always synthetic: 4 * 0.5 = 2.0 -> 4.0 normalized (worse than always-real)
    assert cost_at(y, s, thr=0.0, pi_synth=0.5) == 4.0


def test_min_cost_perfect_and_bayes_threshold():
    import math

    from hearsay.metrics import bayes_llr_threshold, decision_logit, min_cost, sigmoid

    y = np.array([0, 0, 1, 1])
    assert min_cost(y, np.array([-2.0, -1.0, 1.0, 2.0])) == 0.0
    assert math.isclose(bayes_llr_threshold(0.5), math.log(4))
    # posterior 0.8 at pi=0.5 <=> LLR ln4 <=> decision prob 0.5
    assert math.isclose(float(sigmoid(decision_logit(math.log(4), 0.5))), 0.5)
    # above pi_synth = 0.8 the constant decision flips to "synthetic"
    assert sigmoid(decision_logit(0.0, 0.79)) < 0.5 < sigmoid(decision_logit(0.0, 0.81))


def test_normalize_windows_removes_level_and_offset():
    from hearsay.embed import normalize_windows

    rng = np.random.default_rng(0)
    w = rng.standard_normal((2, 1000)).astype(np.float32)
    a = normalize_windows(w * 0.54 + 0.1)
    b = normalize_windows(w * 1.0)
    assert np.allclose(a, b, atol=1e-4)
    assert np.allclose(a.mean(axis=1), 0, atol=1e-5) and np.allclose(a.std(axis=1), 1, atol=1e-3)
    assert np.isfinite(normalize_windows(np.zeros((1, 100)))).all()  # silence stays finite


def test_sponsor_min_dcf_matches_shipped_code():
    import sys

    from hearsay.metrics import sponsor_min_dcf

    pkg = Path(__file__).resolve().parents[1] / "data/nsa/HackGTMinDCF/asvspoof5/evaluation-package"
    if not pkg.exists():
        pytest.skip("sponsor evaluation package not present")
    sys.path.insert(0, str(pkg))
    from calculate_modules import compute_eer, compute_mindcf

    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(700, int), np.ones(300, int)]
    s = np.r_[rng.normal(0.3, 0.15, 700), rng.normal(0.7, 0.15, 300)]
    for flip in (False, True):
        t = -s if flip else s
        _, frr, far, thr, _ = compute_eer(t[y == 0], t[y == 1])
        ref = compute_mindcf(frr, far, thr, 0.5, 1, 4)[0]
        assert abs(sponsor_min_dcf(y, s, flip) - ref) < 1e-9
    assert sponsor_min_dcf(y, s, flip=False) > 0.9 > sponsor_min_dcf(y, s, flip=True)


def test_trim_silence_is_level_independent_and_safe():
    from hearsay.audio import trim_silence

    t = np.arange(SR) / SR
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x = np.concatenate([np.zeros(SR // 2, np.float32), tone, np.zeros(SR // 4, np.float32)])
    y = trim_silence(x)
    assert abs(y.size - SR) <= SR // 50
    assert np.array_equal(trim_silence(x * 0.01).size, y.size)  # same cut at any level
    assert trim_silence(np.zeros(SR, np.float32)).size == SR  # all-silent: unchanged
    short = np.concatenate([np.zeros(SR, np.float32), tone[: SR // 4]])
    assert trim_silence(short).size == short.size  # would drop below 1 s: unchanged


def test_prepare_segment_never_tiles_and_crops_deterministically():
    from hearsay.embed import prepare_segment

    rng = np.random.default_rng(0)
    short = (0.1 * rng.standard_normal(int(3.2 * SR))).astype(np.float32)
    assert prepare_segment(short).size == short.size  # no padding/tiling to 4 s
    assert prepare_segment(short, crop_s=5.0, seed=1).size == short.size  # crop > clip: as is
    long = (0.1 * rng.standard_normal(10 * SR)).astype(np.float32)
    a, b = prepare_segment(long, crop_s=3.4, seed=7), prepare_segment(long, crop_s=3.4, seed=7)
    assert a.size == int(3.4 * SR) and np.array_equal(a, b)
    assert prepare_segment(long).size == 8 * SR  # test-time cap
    assert prepare_segment(long, crop_s=3.4, seed=8).tobytes() != a.tobytes()


def test_test_duration_sampler_matches_test_range():
    from hearsay.embed import TEST_DURATIONS, test_duration_sampler

    if not TEST_DURATIONS.exists():
        pytest.skip("test durations not present")
    draw = test_duration_sampler(0)
    xs = [draw() for _ in range(500)]
    assert 3.0 <= min(xs) and max(xs) <= 14.0 and 3.2 < float(np.median(xs)) < 3.7


def test_prepare_segment_band_matches_to_nsa_wall():
    from hearsay.embed import prepare_segment

    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(4 * SR)).astype(np.float32)  # white: energy up to 8 kHz

    def share_above(y, hz):
        spec = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(y.size, 1 / SR)
        return spec[f >= hz].sum() / spec.sum()

    assert share_above(x, 7500) > 0.05
    y = prepare_segment(x)
    assert share_above(y, 7500) < 1e-4  # the wall is reproduced
    assert share_above(prepare_segment(x, band_match=False), 7500) > 0.05  # counterfactual
    assert abs(share_above(y, 1000) - share_above(x, 1000)) < 0.15  # passband mostly intact
