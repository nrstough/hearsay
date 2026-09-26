"""On-box codec round-trips, exercised with the local ffmpeg (spec appendix B5)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hearsay import SR

CLOUD = Path(__file__).resolve().parents[1] / "scripts" / "cloud"


def _speech_like(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 560, 1100, 2200), 1))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(t.size)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


@pytest.mark.slow
def test_b5_codec_copies_keep_length_alignment_and_label(tmp_path):
    import sys

    sys.path.insert(0, str(CLOUD))
    from box_codecs import available_codecs, roundtrip

    codecs = available_codecs()
    assert "mp3" in codecs and "mulaw" in codecs
    src = tmp_path / "a.flac"
    x = _speech_like(3.0)
    sf.write(src, x, SR, subtype="PCM_16", format="FLAC")
    for c in codecs:
        r = roundtrip(str(src), c, str(tmp_path / f"a.{c}.flac"))
        assert "error" not in r, (c, r)
        assert abs(r["duration"] - r["src_duration"]) <= 0.05 * r["src_duration"] + 0.05, (c, r)
        assert abs(r["lag_ms"]) < 50, (c, r)
        assert r["lead_s"] < 0.03, (c, r)
        y, sr = sf.read(tmp_path / f"a.{c}.flac", dtype="float32")
        assert sr == SR and np.isfinite(y).all() and np.abs(y).max() > 0.05


@pytest.mark.slow
def test_b5_script_selects_class_blind_and_writes_manifest(tmp_path):
    bundle = tmp_path / "b"
    (bundle / "core").mkdir(parents=True)
    (bundle / "extra").mkdir()
    rows = []
    for i in range(40):
        scope = "core" if i < 30 else "extra_mlaad"
        f = f"c{i}.flac"
        sf.write(bundle / ("core" if scope == "core" else "extra") / f, _speech_like(1.5, i), SR,
                 subtype="PCM_16", format="FLAC")
        rows.append({"id": f"c{i}", "file": f, "label": "spoof" if i % 2 else "bonafide",
                     "fold": "holdout" if i % 10 == 0 else str(i % 5), "train_scope": scope})
    pd.DataFrame(rows).to_csv(bundle / "manifest.csv", index=False)
    r = subprocess.run(["uv", "run", "python", str(CLOUD / "box_codecs.py"), "--bundle", str(bundle),
                        "--frac", "0.5", "--workers", "2", "--codecs", "mp3,mulaw"],
                       capture_output=True, text=True, cwd=CLOUD.parents[1], check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    cm = pd.read_csv(bundle / "codecs" / "codec_manifest.csv")
    m = pd.read_csv(bundle / "manifest.csv").set_index("id")
    assert set(m.loc[m.fold == "holdout"].index) <= set(cm.id)  # every holdout clip
    picked = m.loc[cm.id]
    share = picked.label.value_counts(normalize=True)
    assert abs(share.get("spoof", 0) - 0.5) < 0.25  # not label-driven
    assert set(cm.codec) <= {"mp3", "mulaw"}
    assert all((bundle / "codecs" / f).exists() for f in cm.file)
