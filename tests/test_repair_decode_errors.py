"""scripts/repair_decode_errors.py: re-embeds only decode_error rows, through the extractor's
crop and seed, after an identity check on unflagged rows. Hermetic: temp shards and a fake
embedder; no audio, weights or ffmpeg."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _load("repair_decode_errors")
DecodeError = rep.DecodeError
SEED0 = 100


def fake_embed(model, path, crop_s, seed):
    """Deterministic in (path, crop, seed); records its calls."""
    fake_embed.calls.append((path, None if np.isnan(crop_s) else round(crop_s, 3), seed))
    v = (int(path.split("_")[1]) * 7 + seed * 3 + (0 if np.isnan(crop_s) else crop_s)) % 97
    return np.full((2, 3), v + 1.0, dtype=np.float16)


fake_embed.calls = []


def _make(tmp: Path, n=10, shard=4, flagged=(2, 5, 9), mode="segment", crop=True):
    out = tmp / "set"
    out.mkdir()
    (out / "extract_meta.json").write_text(json.dumps({"mode": mode, "seed": SEED0}))
    man = pd.DataFrame({"path": [f"clip_{i}" for i in range(n)], "label": "bonafide"})
    crops = np.array([2.5 + 0.1 * i if crop else np.nan for i in range(n)])
    for s0 in range(0, n, shard):
        rows = np.arange(s0, min(s0 + shard, n))
        emb = np.stack([fake_embed(None, man.path[r], crops[r], SEED0 + r) for r in rows])
        flag = np.array(["decode_error" if r in flagged else "" for r in rows])
        emb[flag == "decode_error"] = 0
        np.savez(out / f"shard_{s0 // shard:05d}.npz", emb=emb, row=rows,
                 n_windows=np.ones(len(rows), int), flag=flag, crop_s=crops[rows])  # fmt: skip
    man.assign(n_windows=1, flag=["decode_error" if r in flagged else "" for r in range(n)]).to_csv(
        out / "manifest.csv", index=False)  # fmt: skip
    fake_embed.calls.clear()
    return out, man


def _all(out: Path):
    zs = [np.load(p) for p in sorted(out.glob("shard_*.npz"))]
    return (np.concatenate([z["emb"] for z in zs]), np.concatenate([z["flag"] for z in zs]))


def test_fixes_only_flagged_rows_with_stored_crop_and_row_seed(tmp_path):
    out, man = _make(tmp_path)
    before, _ = _all(out)
    res = rep.repair(None, out, man, n_verify=2, embed=fake_embed)
    after, flags = _all(out)
    assert sorted(res["fixed"]) == [2, 5, 9] and res["still_failing"] == []
    assert (flags == "").all()
    for r in [2, 5, 9]:
        assert np.array_equal(after[r], fake_embed(None, f"clip_{r}", 2.5 + 0.1 * r, SEED0 + r))
        assert after[r].any()
    untouched = [r for r in range(10) if r not in (2, 5, 9)]
    assert np.array_equal(after[untouched], before[untouched])
    repaired_calls = [c for c in fake_embed.calls if c[0] in ("clip_2", "clip_5", "clip_9")]
    assert ("clip_5", 3.0, SEED0 + 5) in repaired_calls
    assert (pd.read_csv(out / "manifest.csv").flag.fillna("") == "").all()


def test_uncropped_rows_pass_nan_crop(tmp_path):
    out, man = _make(tmp_path, crop=False)
    rep.repair(None, out, man, n_verify=1, embed=fake_embed)
    assert all(c[1] is None for c in fake_embed.calls)


def test_identity_mismatch_refuses_and_writes_nothing(tmp_path):
    out, man = _make(tmp_path)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def drifted(model, path, crop_s, seed):
        return fake_embed(model, path, crop_s, seed) + np.float16(5)

    with pytest.raises(AssertionError, match="does not reproduce"):
        rep.repair(None, out, man, n_verify=2, embed=drifted)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_row_that_still_fails_stays_flagged(tmp_path):
    out, man = _make(tmp_path)

    def flaky(model, path, crop_s, seed):
        if path == "clip_5":
            raise DecodeError("still broken")
        return fake_embed(model, path, crop_s, seed)

    res = rep.repair(None, out, man, n_verify=2, embed=flaky)
    _, flags = _all(out)
    assert res["still_failing"] == [5] and sorted(res["fixed"]) == [2, 9]
    assert flags[5] == "decode_error" and (np.delete(flags, 5) == "").all()
    assert pd.read_csv(out / "manifest.csv").flag.fillna("").tolist()[5] == "decode_error"


def test_no_flagged_rows_is_a_noop(tmp_path):
    out, man = _make(tmp_path, flagged=())
    before, _ = _all(out)
    res = rep.repair(None, out, man, n_verify=2, embed=fake_embed)
    assert res["fixed"] == [] and np.array_equal(_all(out)[0], before)


def test_windows_mode_refused(tmp_path):
    out, man = _make(tmp_path, mode="windows")
    with pytest.raises(AssertionError, match="segment-mode"):
        rep.repair(None, out, man, embed=fake_embed)
