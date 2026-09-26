"""scripts/repair_decode_errors.py: re-embeds only decode_error rows, through the extractor's
crop and seed, after an element-wise identity check on unflagged rows; --verify-only checks a
finished set against its twin. Hermetic: temp shards, a fake embedder and a stub model; no
real audio, weights or ffmpeg."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

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


def test_verified_rows_are_unflagged_and_reported(tmp_path):
    out, man = _make(tmp_path)
    res = rep.repair(None, out, man, n_verify=3, embed=fake_embed)
    assert len(res["verified_rows"]) == 3 and not set(res["verified_rows"]) & {2, 5, 9}
    assert res["verify_worst_excess"] <= 0 and res["verify_max_abs_diff"] == 0.0


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


def test_empty_verify_pool_refuses(tmp_path):
    out, man = _make(tmp_path, n=4, shard=4, flagged=(0, 1, 2, 3))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(AssertionError, match="unflagged rows to verify"):
        rep.repair(None, out, man, n_verify=1, embed=fake_embed)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_zero_verify_refuses(tmp_path):
    out, man = _make(tmp_path)
    with pytest.raises(AssertionError, match="at least one row"):
        rep.repair(None, out, man, n_verify=0, embed=fake_embed)


def test_verify_pool_falls_back_to_other_shards(tmp_path):
    # the only flagged shard is fully flagged; unflagged rows elsewhere still verify
    out, man = _make(tmp_path, n=8, shard=4, flagged=(4, 5, 6, 7))
    res = rep.repair(None, out, man, n_verify=2, embed=fake_embed)
    assert sorted(res["fixed"]) == [4, 5, 6, 7] and all(r < 4 for r in res["verified_rows"])


def test_manifest_mismatch_refuses(tmp_path):
    out, man = _make(tmp_path)
    wrong = man.assign(path=man.path[::-1].to_numpy())
    with pytest.raises(AssertionError, match="same paths"):
        rep.repair(None, out, wrong, embed=fake_embed)


def test_unfinished_set_refuses(tmp_path):
    out, man = _make(tmp_path)
    (out / "manifest.csv").unlink()
    with pytest.raises(AssertionError, match="has not finished"):
        rep.repair(None, out, man, embed=fake_embed)
    (tmp_path / "b").mkdir()
    out2, man2 = _make(tmp_path / "b")
    max(out2.glob("shard_*.npz")).unlink()
    with pytest.raises(AssertionError, match="cover every manifest row"):
        rep.repair(None, out2, man2, embed=fake_embed)


def test_tolerance_is_element_wise_at_wavlm_scale():
    stored = np.full((25, 1024), 1.0, dtype=np.float16)
    stored[0, 0] = 500.0  # one large element must not license a large error elsewhere
    drift = stored.copy()
    drift[5, 5] += np.float16(0.5)
    assert rep.excess(stored.copy(), stored) <= 0
    assert rep.excess(drift, stored) > 0
    rounding = (stored.astype(np.float32) * (1 + 1e-3)).astype(np.float16)
    assert rep.excess(rounding, stored) <= 0


def test_no_tmp_file_left_after_repair(tmp_path):
    out, man = _make(tmp_path)
    rep.repair(None, out, man, n_verify=2, embed=fake_embed)
    assert not list(out.glob("*.tmp.npz"))


class _StubModel(torch.nn.Module):
    """2 identical 'hidden states' of 3 features: the input's mean, its length in s, its peak."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, output_hidden_states=True):
        n = x.shape[1]
        feats = torch.stack(
            [x.mean(1), torch.full_like(x[:, 0], n / 16000), x.abs().max(1).values], 1
        )
        return type("O", (), {"hidden_states": [feats[:, None, :]] * 2})()


def test_embed_row_runs_the_real_segment_path(monkeypatch):
    rng = np.random.default_rng(0)
    sig = (0.3 * np.sin(np.arange(6 * 16000) * 2 * np.pi * 220 / 16000)
           * (1 + 0.5 * rng.standard_normal(6 * 16000))).astype(np.float32)  # fmt: skip
    monkeypatch.setattr(rep, "load_audio", lambda path: sig)
    full = rep.embed_row(_StubModel(), "x", float("nan"), 0)
    cropped = rep.embed_row(_StubModel(), "x", 2.0, 7)
    again = rep.embed_row(_StubModel(), "x", 2.0, 7)
    assert full.dtype == np.float16 and full.shape == (2, 3)
    assert full[0, 1] > 4.5 and cropped[0, 1] == 2.0  # NaN = no crop; a 2 s crop
    assert np.array_equal(cropped, again)  # the row seed fixes the crop offset
    expected = rep.embed_segment(_StubModel(), rep.prepare_segment(sig, 2.0, 7)).astype(np.float16)
    assert np.array_equal(cropped, expected)


def _twin(tmp: Path, out: Path, crop=True):
    tw = tmp / "twin"
    tw.mkdir()
    (tw / "extract_meta.json").write_text((out / "extract_meta.json").read_text())
    (tw / "manifest.csv").write_text((out / "manifest.csv").read_text())
    for p in out.glob("shard_*.npz"):
        z = dict(np.load(p))
        if not crop:
            z["crop_s"] = z["crop_s"] + 0.5
        np.savez(tw / p.name, **z)
    return tw


def test_verify_only_passes_on_a_clean_set_and_writes_nothing(tmp_path):
    out, man = _make(tmp_path, flagged=())
    tw = _twin(tmp_path, out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    res = rep.verify_set(None, out, man, tw, per_shard=2, rows=(7,), embed=fake_embed)
    assert (
        res["ok"] and res["meta_equal_twin"] and res["paths_equal_twin"] and res["crop_equal_twin"]
    )
    assert res["n_identity_checked"] >= 6 and 7 in res["explicit_rows_checked"]
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_verify_only_fails_on_flags_crops_or_drift(tmp_path):
    out, man = _make(tmp_path)  # rows 2, 5, 9 still flagged
    assert not rep.verify_set(None, out, man, per_shard=1, embed=fake_embed)["ok"]
    (tmp_path / "b").mkdir()
    out2, man2 = _make(tmp_path / "b", flagged=())
    res = rep.verify_set(None, out2, man2, _twin(tmp_path / "b", out2, crop=False), per_shard=1,
                         embed=fake_embed)  # fmt: skip
    assert not res["crop_equal_twin"] and not res["ok"]

    def drifted(model, path, crop_s, seed):
        return fake_embed(model, path, crop_s, seed) + np.float16(5)

    assert not rep.verify_set(None, out2, man2, per_shard=1, embed=drifted)["ok"]
