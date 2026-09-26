"""M5 bundle writer (spec appendix B1-B8, minus B5 which lives in test_m5_codecs)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hearsay import SR
from hearsay.audio import load_audio, trim_silence
from hearsay.embed import normalize_windows
from hearsay.m5_bundle import clip_ok, prepare_clip, read_clip, tree_sha, write_clip

REPO = Path(__file__).resolve().parents[1]


def ffmpeg_sine(out: Path, *, sr: int = 44_100, ch: int = 2, dur: float = 1.0, args=()) -> Path:
    src = f"sine=frequency=440:sample_rate={sr}:duration={dur}"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-ac", str(ch)]
    subprocess.run([*cmd, *args, str(out)], check=True)
    return out


@pytest.mark.parametrize(
    ("name", "sr", "ch", "args"),
    [
        ("int16.wav", 16_000, 1, ()),
        ("f32.wav", 16_000, 1, ("-c:a", "pcm_f32le")),
        ("stereo22k.wav", 22_050, 2, ()),
        ("cbr.mp3", 44_100, 2, ("-b:a", "64k")),
        ("lossless.flac", 48_000, 1, ()),
        ("aac.mp4", 48_000, 2, ("-c:a", "aac", "-f", "mp4")),
        ("opus.ogg", 48_000, 1, ("-c:a", "libopus")),
    ],
)
def test_b1_every_input_format_lands_as_16k_mono_pcm16_flac(tmp_path, name, sr, ch, args):
    src = ffmpeg_sine(tmp_path / name, sr=sr, ch=ch, dur=2.0, args=args)
    r = write_clip(src, tmp_path / "out.flac")
    info = sf.info(str(tmp_path / "out.flac"))
    assert info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16"
    assert abs(r["duration"] - 2.0) < 0.15  # trim + codec padding
    assert 0.05 < r["peak"] <= 0.999 and len(r["pcm_sha256"]) == 64


def test_b2_bundled_clip_equals_trim_silence_of_load_audio(tmp_path):
    src = ffmpeg_sine(tmp_path / "a.wav", sr=SR, ch=1, dur=3.0)
    write_clip(src, tmp_path / "a.flac")
    ref = trim_silence(load_audio(src))
    got = read_clip(tmp_path / "a.flac")
    assert got.size == ref.size
    assert np.abs(got - ref).max() < 1.5 / 32768


def test_b3_over_range_input_is_peak_scaled_and_normalization_unchanged():
    rng = np.random.default_rng(0)
    x = (1.5 * rng.standard_normal(3 * SR)).astype(np.float32)
    y = prepare_clip(x)
    assert np.abs(y).max() <= 0.999 + 1e-6
    assert np.allclose(normalize_windows(x[None])[0], normalize_windows(y[None])[0], atol=1e-4)


def test_b4_length_cap_and_short_rule(tmp_path):
    long = ffmpeg_sine(tmp_path / "long.wav", sr=SR, ch=1, dur=17.0)
    r = write_clip(long, tmp_path / "long.flac")
    assert abs(r["duration"] - 15.0) < 1e-3
    short = ffmpeg_sine(tmp_path / "short.wav", sr=SR, ch=1, dur=0.3)
    assert write_clip(short, tmp_path / "s_extra.flac", min_s=0.5) == {"dropped": "short",
                                                                        "duration": 0.3}
    assert not (tmp_path / "s_extra.flac").exists()
    r = write_clip(short, tmp_path / "s_core.flac")  # core rows: no min_s, never dropped
    assert (tmp_path / "s_core.flac").exists() and r["duration"] == 0.3


def test_b6_decode_error_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF garbage" * 100)
    r = write_clip(bad, tmp_path / "bad.flac")
    assert "error" in r and not (tmp_path / "bad.flac").exists()


def test_b7_tree_sha_covers_every_file_and_clip_ok_gates_reuse(tmp_path):
    src = ffmpeg_sine(tmp_path / "a.wav", sr=SR, ch=1, dur=1.0)
    root = tmp_path / "v1"
    write_clip(src, root / "core" / "a.flac")
    (root / "manifest.csv").write_text("id\na\n")
    s1 = tree_sha(root)
    (root / "manifest.csv").write_text("id\nb\n")  # a manifest edit changes the identity
    s2 = tree_sha(root)
    assert s1 != s2
    (root / "TREE_SHA").write_text(s2)  # the sha file itself is excluded
    assert tree_sha(root) == s2
    (root / "codecs").mkdir()  # box-made variants and the upload marker never change the identity
    (root / "codecs" / "x.flac").write_bytes(b"zz")
    (root / "UPLOAD_DONE").write_text("done")
    assert tree_sha(root) == s2
    assert clip_ok(root / "core" / "a.flac")
    assert clip_ok(root / "core" / "a.flac", expected_frames=SR)
    assert not clip_ok(root / "core" / "a.flac", expected_frames=SR + 1)
    (root / "core" / "a.flac").write_bytes(b"not a flac")
    assert not clip_ok(root / "core" / "a.flac")
    assert not clip_ok(root / "core" / "missing.flac")


@pytest.mark.slow
def test_b8_and_core_abort_smoke_via_script(tmp_path):
    """End to end on a tiny synthetic manifest: test rows keep nsa_test order; a corrupt core
    row aborts; a corrupt extra row is listed in errors.csv."""
    wav = ffmpeg_sine(tmp_path / "ok.wav", sr=SR, ch=1, dur=2.0)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF" * 50)
    rows = []
    for i in range(6):
        rows.append((f"c{i}", str(wav), "spoof" if i % 2 else "bonafide", "g" if i % 2 else "bonafide",
                     f"s{i}", "diffssd" if i % 2 else "librispeech", f"grp{i}", str(i % 5),
                     "core", "g"))  # fmt: skip
    rows.append(("x1", str(bad), "spoof", "M", "s", "mlaad", "mlaad:M", "extra", "extra_mlaad", "M"))
    cols = ["id", "path", "label", "generator", "speaker", "source", "group", "fold",
            "train_scope", "model_name"]  # fmt: skip
    man = tmp_path / "m5_manifest.csv"
    pd.DataFrame(rows, columns=cols).to_csv(man, index=False)
    tm = tmp_path / "nsa_test.csv"
    tw = ffmpeg_sine(tmp_path / "t.wav", sr=SR, ch=1, dur=1.5)
    tw2 = ffmpeg_sine(tmp_path / "t2.wav", sr=SR, ch=1, dur=1.2)
    pd.DataFrame({"filename": ["b.wav", "a.wav"], "path": [str(tw), str(tw2)]}).to_csv(tm, index=False)
    out = tmp_path / "bundle"
    cmd = [sys.executable, str(REPO / "scripts/m5_build_bundle.py"), "--manifest", str(man),
           "--test-manifest", str(tm), "--out", str(out), "--workers", "2",
           "--max-error-frac", "1.0"]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    t = pd.read_csv(out / "test" / "manifest.csv")
    assert t.filename.tolist() == ["b.wav", "a.wav"]  # nsa_test order kept
    errs = pd.read_csv(out / "errors.csv")
    assert errs.id.tolist() == ["x1"]
    m = pd.read_csv(out / "manifest.csv")
    assert "x1" not in set(m.id) and len(m) == 6
    assert (out / "TREE_SHA").read_text() == tree_sha(out)
    assert len(pd.read_csv(out / "test_overlap.csv")) == 0
    # a corrupt CORE row must abort
    rows[0] = ("c0", str(bad), *rows[0][2:])
    pd.DataFrame(rows, columns=cols).to_csv(man, index=False)
    r = subprocess.run(cmd + ["--out", str(tmp_path / "b2")], capture_output=True, text=True, cwd=REPO, check=False)
    assert r.returncode != 0 and "core row" in r.stderr


def test_b7_changed_source_gets_a_new_stamp(tmp_path):
    """A resumed build reuses a FLAC only if the source's stamp (size:mtime) is unchanged."""
    import os
    import time

    from hearsay.m5_bundle import source_stamp

    src = ffmpeg_sine(tmp_path / "a.wav", sr=SR, ch=1, dur=1.0)
    write_clip(src, tmp_path / "a.flac")
    stamp1 = source_stamp(src)
    ffmpeg_sine(src, sr=SR, ch=1, dur=2.0)  # new content at the same path
    os.utime(src, (time.time() + 5, time.time() + 5))
    assert source_stamp(src) != stamp1
    build = (REPO / "scripts" / "m5_build_bundle.py").read_text()
    assert "stamps.get(dst.stem) == source_stamp(src)" in build  # the reuse gate in the build
