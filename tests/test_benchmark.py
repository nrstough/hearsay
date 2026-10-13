"""Tests for the V2-01-lite benchmark (src/hearsay/benchmark.py, scripts/v2_benchmark.py).

Run spec: docs/specs/2026-10-11_v2-benchmark-fusion.md (sections L, S1-S8).
Hermetic tests build a small seeded world under tmp_path. Tests marked
needs_data read the real exports and skip themselves when they are absent.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from hearsay import benchmark as B
from hearsay.metrics import sponsor_min_dcf

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v2_benchmark", REPO / "scripts/v2_benchmark.py")
VB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VB)

HAVE_DATA = (REPO / "outputs/detector_scores/m1b_v3.csv").exists() and (REPO / "splits/nsa_folds.csv").exists()
def needs_data(f):
    """Marker (selectable with -m needs_data) plus a skip when the exports are absent."""
    return pytest.mark.needs_data(pytest.mark.skipif(not HAVE_DATA, reason="needs the gitignored score exports")(f))
TRACKED_LEDGER = REPO / "benchmarks/holdout_ledger.jsonl"


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    """Every test uses a tmp holdout ledger; the tracked ledger must not change."""
    before = TRACKED_LEDGER.read_bytes() if TRACKED_LEDGER.exists() else None
    monkeypatch.setenv("HEARSAY_HOLDOUT_LEDGER", str(tmp_path / "ledger"))
    yield
    after = TRACKED_LEDGER.read_bytes() if TRACKED_LEDGER.exists() else None
    assert before == after, "a test changed the tracked holdout ledger"


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------


def make_world(tmp_path: Path, seed: int = 0):
    """5 inner folds + holdout, LJ + LibriSpeech reals, 2 generators per fold
    (1 in folds 3 and 4, as in the real data), 60 ITW rows, 10 test rows."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "repo"
    for d in ("splits", "outputs/manifests"):
        (root / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in ["0", "1", "2", "3", "4", "holdout"]:
        gens = [f"g{fold}a"] if fold in ("3", "4") else [f"g{fold}a", f"g{fold}b"]
        for g in gens:
            for i in range(20):
                rows.append((f"{root}/data/nsa/{g}/clip_{i}.wav", "spoof", g, f"spk{i % 4}", "diffssd", f"gen:{g}", fold))
        for i in range(16):
            rows.append((f"{root}/data/ljspeech/wavs/LJ0{fold}{i}.wav", "bonafide", "bonafide", "LJ", "ljspeech",
                         f"lj:LJ0{fold}", fold))
        for s in range(4):
            for i in range(6):
                spk = f"libri_{fold}{s}"
                rows.append((f"{root}/data/librispeech/{spk}/{i}.flac", "bonafide", "bonafide", spk, "librispeech",
                             f"spk:{spk}", fold))
    folds = pd.DataFrame(rows, columns=["path", "label", "generator", "speaker", "source", "group", "fold"])
    folds.to_csv(root / "splits/nsa_folds.csv", index=False)
    itw = pd.DataFrame({
        "path": [f"data/in_the_wild/release_in_the_wild/{i}.wav" for i in range(60)],
        "label": ["spoof" if i % 3 == 0 else "bonafide" for i in range(60)],
        "generator": ["itw_spoof" if i % 3 == 0 else "bonafide" for i in range(60)],
        "speaker": [f"person{i % 6}" for i in range(60)],
        "source": "in_the_wild",
    })
    itw.to_csv(root / "outputs/manifests/itw_stress.csv", index=False)
    test = pd.DataFrame({"filename": [f"HGT{i}.wav" for i in range(10)],
                         "path": [f"data/nsa/HackGTHearsayTesting/HGT{i}.wav" for i in range(10)]})
    test.to_csv(root / "outputs/manifests/nsa_test.csv", index=False)
    pops = B.load_populations(root, main=root, expect=None)
    return root, pops, rng


def write_scores(root: Path, pops, name: str, rng, *, scale=2.0, sign=1.0, splits=("inner", "holdout", "itw"),
                 drop=0, relative=False, with_score=True) -> Path:
    parts = []
    for p in splits:
        pop = pops[p]
        y = pop.y if pop.y is not None else np.zeros(pop.n)
        logit = sign * (scale * (y - 0.5)) + rng.normal(0, 1, pop.n)
        keys = pop.keys
        paths = [k if (relative or not k.startswith("data/nsa/g") and not k.startswith("data/l")) else f"{root}/{k}"
                 for k in keys]
        fold = pop.fold if pop.fold is not None else [""] * pop.n
        parts.append(pd.DataFrame({"path": paths, "fold": fold, "split": B.POP_SPLIT[p], "logit": logit}))
    df = pd.concat(parts, ignore_index=True)
    if drop:
        df = df.iloc[drop:]
    if with_score:
        df["score"] = expit(df["logit"])
    out = root / "scores" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def read(path, root, **kw):
    return B.read_score_file(path, main=root, repo=root, **kw)


@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path)


def bench_with(world, names=("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist"), splits=("inner", "holdout", "itw")):
    root, pops, rng = world
    bench = B.Bench(pops)
    for n in names:
        bench.add(read(write_scores(root, pops, n, rng, splits=splits), root, name=n))
    return bench


# ---------------------------------------------------------------------------
# L. Leakage
# ---------------------------------------------------------------------------


def test_nested_refs_exclude_scored_fold():
    """A row's nested rank depends only on its own value and the OTHER folds:
    changing every other row of its own fold must not move it."""
    rng = np.random.default_rng(1)
    logit = rng.normal(size=200)
    fold = np.array([str(i % 5) for i in range(200)])
    r0 = B.nested_ranks(logit, fold)
    i = 0
    poisoned = logit.copy()
    same = (fold == fold[i]) & (np.arange(200) != i)
    poisoned[same] = logit[i] - 100.0  # every same-fold row now sits below row i
    r1 = B.nested_ranks(poisoned, fold)
    assert r1[i] == r0[i]
    ref = np.sort(logit[fold != fold[i]])
    assert r0[i] == np.searchsorted(ref, logit[i], side="left") / len(ref)
    trans = B.full_ranks(logit, logit)
    assert not np.allclose(trans, r0)


def test_select_refuses_report_only_population():
    for role in ("read_once", "report_only", "in_sample", "reference"):
        with pytest.raises(B.BenchmarkError):
            B.require_selectable("some_pop", role)
    B.require_selectable("inner", "selectable")


def test_mlaad_role_in_sample_for_m5():
    with pytest.raises(B.BenchmarkError, match="in-sample for M5"):
        B.require_selectable("mlaad", "selectable", rule_columns=("m1b_v3", "m5_xlsr_ft"))


def test_holdout_read_ledger_refuses_second_read(tmp_path):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    calls = []

    def compute():
        calls.append(1)
        return {"x": 1.0}

    defs = [{"kind": "fixed", "weights": {"a": 1.0}, "e": True}]
    r1 = B.read_holdout(defs, compute, ledger=led, holdout_sha="h", meta={"code_sha": "c", "input_shas": {}})
    r2 = B.read_holdout(defs, compute, ledger=led, holdout_sha="h", meta={"code_sha": "c", "input_shas": {}})
    assert len(calls) == 1
    assert r2["readout"] == r1["readout"] and r2["ledger"].startswith("persisted")
    # a crashed reservation under a different code state is refused
    key = B.ledger_key([{"kind": "x"}], "h")
    led.reserve(key, {"code_sha": "c1", "input_shas": {}})
    with pytest.raises(B.HoldoutAlreadyRead):
        led.reserve(key, {"code_sha": "c2", "input_shas": {}})
    assert led.reserve(key, {"code_sha": "c1", "input_shas": {}})["status"] == "resumed"


def test_holdout_ledger_allows_new_candidate_set(tmp_path):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    n = []
    B.read_holdout([{"kind": "fixed", "weights": {"a": 1.0}, "e": True}], lambda: n.append(1) or {}, ledger=led,
                   holdout_sha="h")
    B.read_holdout([{"kind": "fixed", "weights": {"b": 1.0}, "e": True}], lambda: n.append(1) or {}, ledger=led,
                   holdout_sha="h")
    assert len(n) == 2 and led.count() == 2


def test_loader_refuses_asv19_eval_paths(tmp_path):
    for p in ("data/asvspoof2019/LA/ASVspoof2019_LA_eval/flac/LA_E_1.flac",
              "outputs/embeddings/wav2vec2-xls-r-300m/asv19_eval_probe/x.npy",
              f"{tmp_path}/data/x/LA_E_2.flac"):
        with pytest.raises(B.BenchmarkError, match="LA eval"):
            B.canonical_key(p, main=tmp_path, repo=tmp_path)


def test_basis_mismatch_raises(world):
    """A file on a larger inner basis (e.g. 23,790 rows) is refused, not trimmed."""
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng)
    df = pd.read_csv(p)
    extra = df[df["split"] == "inner_oof"].iloc[:5].copy()
    extra["path"] = [f"data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_{i}.flac" for i in range(5)]
    pd.concat([df, extra]).to_csv(p, index=False)
    with pytest.raises(B.BenchmarkError, match="coverage mismatch"):
        B.Bench(pops).add(read(p, root))


@needs_data
def test_inner_basis_is_16142_and_l20_filtered_by_path():
    sf = B.read_score_file(VB.L20_CSV, name="xlsr_L20")
    pops = B.load_populations()
    x = B.align(sf, pops["inner"])
    assert len(x) == 16142
    B.check_inner_folds(sf, pops["inner"])


# ---------------------------------------------------------------------------
# S1. Loader and score-file contract
# ---------------------------------------------------------------------------


def test_canonical_key_abs_and_rel_forms_agree(tmp_path):
    rel = "data/ljspeech/wavs/LJ001-0001.wav"
    assert B.canonical_key(rel, main=tmp_path, repo=tmp_path) == rel
    assert B.canonical_key(f"{tmp_path}/{rel}", main=tmp_path, repo=tmp_path) == rel
    assert B.canonical_key(f"{tmp_path}/data/./ljspeech/wavs/../wavs/LJ001-0001.wav", main=tmp_path, repo=tmp_path) == rel


def test_canonical_key_from_worktree_and_main_agree(tmp_path):
    rel = "data/nsa/x/clip.wav"
    main = tmp_path / "hearsay"
    wt = main / ".claude/worktrees/lane-a"
    assert B.canonical_key(f"{main}/{rel}", main=main, repo=wt) == rel
    assert B.canonical_key(f"{wt}/{rel}", main=main, repo=wt) == rel
    assert B.canonical_key(f"{main}/.claude/worktrees/other/{rel}", main=main, repo=wt) == rel


def test_canonical_key_never_resolves_symlinks(tmp_path):
    target = tmp_path / "volume" / "LJSpeech"
    target.mkdir(parents=True)
    (tmp_path / "repo" / "data").mkdir(parents=True)
    os.symlink(target, tmp_path / "repo" / "data" / "ljspeech")
    root = tmp_path / "repo"
    key = B.canonical_key(f"{root}/data/ljspeech/a.wav", main=root, repo=root)
    assert key == "data/ljspeech/a.wav"
    assert B.canonical_key("data/ljspeech/a.wav", main=root, repo=root) == key


def test_path_outside_repo_raises(tmp_path):
    with pytest.raises(B.BenchmarkError, match="outside the repo"):
        B.canonical_key("/Volumes/Crucial/hearsay/LJSpeech/a.wav", main=tmp_path, repo=tmp_path)
    with pytest.raises(B.BenchmarkError, match="escapes"):
        B.canonical_key("../elsewhere/a.wav", main=tmp_path, repo=tmp_path)


def test_coverage_assert_after_join_raises_on_one_missing_row(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng, drop=1)
    with pytest.raises(B.BenchmarkError, match="1 missing"):
        B.Bench(pops).add(read(p, root))


def test_duplicate_path_raises(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng)
    df = pd.read_csv(p)
    pd.concat([df, df.iloc[:1]]).to_csv(p, index=False)
    with pytest.raises(B.BenchmarkError, match="duplicate"):
        read(p, root)


def test_nsa_join_uses_full_path_not_basename(world):
    """clip_0.wav exists under many generators: each keeps its own score."""
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng)
    sf = read(p, root)
    x = B.align(sf, pops["inner"])
    s = sf.split("inner_oof")
    names = pd.Series([Path(k).name for k in pops["inner"].keys])
    assert names.duplicated().any()
    assert np.array_equal(x, s.reindex(pops["inner"].keys).to_numpy())
    assert len(set(np.round(x[names == "clip_0.wav"], 9))) > 1


def test_unknown_split_raises(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng)
    df = pd.read_csv(p)
    df.loc[0, "split"] = "train"
    df.to_csv(p, index=False)
    with pytest.raises(B.BenchmarkError, match="unknown split"):
        read(p, root)


def test_stress_maps_to_itw(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng)
    df = pd.read_csv(p)
    df["split"] = df["split"].replace({"itw": "stress"})
    df.to_csv(p, index=False)
    sf = read(p, root)
    assert B.align(sf, pops["itw"]) is not None


def test_fold_coerced_to_str():
    assert B._fold_str(4.0) == "4" and B._fold_str("4.0") == "4" and B._fold_str(3) == "3"
    assert B._fold_str("holdout") == "holdout" and B._fold_str(float("nan")) == ""


def test_nonfinite_logit_raises(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng, with_score=False)
    df = pd.read_csv(p)
    df.loc[3, "logit"] = np.inf
    df.to_csv(p, index=False)
    with pytest.raises(B.BenchmarkError, match="non-finite"):
        read(p, root)


def test_direction_guard_inner_auc_below_half_raises(world):
    root, pops, rng = world
    p = write_scores(root, pops, "inv", rng, sign=-1.0)
    with pytest.raises(B.BenchmarkError, match="AUC"):
        B.Bench(pops).add(read(p, root))


def test_reader_uses_logit_column(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng)
    df = pd.read_csv(p)
    sf = read(p, root)
    assert np.array_equal(sf.df["logit"].to_numpy(), df["logit"].to_numpy())


def test_score_logit_sigmoid_consistency_checked(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng)
    df = pd.read_csv(p)
    df.loc[0, "score"] = 0.5 if df.loc[0, "logit"] > 1 else 0.99
    df.to_csv(p, index=False)
    with pytest.raises(B.BenchmarkError, match="sigmoid"):
        read(p, root)


@needs_data
def test_itw_override_sources_pinned():
    assert VB.COLUMNS["handcrafted_v5"]["itw"] == "outputs/detector_scores/_itw_handcrafted_v5.csv"
    assert VB.COLUMNS["m5_xlsr_ft"]["itw"] == "outputs/detector_scores/_itw_m5.csv"
    assert VB.COLUMNS["spectra_aasist"]["itw"] == "outputs/spectra/itw_stress/scores.csv"
    assert VB.COLUMNS["m1b_v3"]["itw"] is None and VB.COLUMNS["wavlm_l"]["itw"] is None


def test_resolve_itw_missing_override_raises(world):
    """No silent fallback: a named override that does not exist is an error, and
    a file without ITW rows is 'not scored' on ITW, never filled from elsewhere."""
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng, splits=("inner", "holdout"))
    with pytest.raises(FileNotFoundError):
        read(p, root, itw=root / "nope.csv")
    assert B.align(read(p, root), pops["itw"]) is None


def test_itw_override_schema_accepts_path_logit_and_stress(world):
    root, pops, rng = world
    p = write_scores(root, pops, "c", rng, splits=("inner", "holdout"))
    o = root / "itw.csv"
    pd.DataFrame({"path": pops["itw"].keys, "logit": rng.normal(size=pops["itw"].n), "split": "stress",
                  "fold": "stress", "flag": np.nan}).to_csv(o, index=False)
    sf = read(p, root, itw=o)
    assert B.align(sf, pops["itw"]) is not None
    assert set(sf.extra_columns) == {"fold", "flag"}
    bad = root / "bad.csv"
    pd.DataFrame({"path": pops["itw"].keys, "logit": 0.0, "split": "holdout"}).to_csv(bad, index=False)
    with pytest.raises(B.BenchmarkError, match="split values"):
        read(p, root, itw=bad)
    with pytest.raises(B.BenchmarkError, match="own itw rows"):
        read(write_scores(root, pops, "d", rng), root, itw=o)


def test_schema_missing_column_raises(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"path": ["data/a.wav"], "score": [0.5]}).to_csv(p, index=False)
    with pytest.raises(B.BenchmarkError, match="missing required"):
        B.read_score_file(p, main=tmp_path, repo=tmp_path)


def test_contract_example_file_round_trips(world):
    """The minimal Interface v1 file (path, split, logit only) reads and joins."""
    root, pops, rng = world
    rows = []
    for p in ("inner", "holdout"):
        for k, y in zip(pops[p].keys, pops[p].y):
            rows.append({"path": k, "split": B.POP_SPLIT[p], "logit": float(2 * y - 1 + rng.normal())})
    f = root / "contract.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    bench = B.Bench(pops)
    bench.add(read(f, root, name="contract"))
    assert bench.cols["contract"]["itw"] is None and len(bench.cols["contract"]["inner"]) == pops["inner"].n


# ---------------------------------------------------------------------------
# S2. Registry
# ---------------------------------------------------------------------------


def test_sha256_mismatch_refuses(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n")
    reg = {"inputs": {"a.csv": {"kind": "file", "sha256": B.sha256_file(f)}}}
    assert B.check_registry(reg, tmp_path) == []
    f.write_text("x\n2\n")
    assert B.check_registry(reg, tmp_path) == ["sha256 mismatch a.csv"]
    (tmp_path / "d").mkdir()
    (tmp_path / "d/w.bin").write_bytes(b"1")
    reg = {"inputs": {"d": {"kind": "tree", "sha256": B.sha256_tree(tmp_path / "d")}}}
    (tmp_path / "d/w.bin").write_bytes(b"2")
    assert B.check_registry(reg, tmp_path) == ["sha256 mismatch (tree) d"]


def test_every_population_has_known_role(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(json.dumps({"populations": {"x": {"role": "selectable"}}}))
    B.load_registry(p)
    p.write_text(json.dumps({"populations": {"x": {"role": "sometimes"}}}))
    with pytest.raises(B.BenchmarkError, match="unknown role"):
        B.load_registry(p)
    for name, meta in VB.POPULATIONS.items():
        assert meta["role"] in B.ROLES, name


@needs_data
def test_registry_counts_match():
    reg = B.load_registry(VB.REGISTRY)
    pops = B.load_populations()
    for k in ("inner", "holdout", "itw", "test"):
        assert pops[k].n == reg["populations"][k]["rows"]
    assert reg["populations"]["holdout_cohort"]["rows"] == len(pd.read_csv(REPO / "outputs/channel/m3_perturb.csv"))
    assert reg["populations"]["mlaad"]["rows"] == len(pd.read_csv(REPO / "outputs/channel/m3_mlaad.csv"))


@needs_data
def test_deferred_populations_absent_and_listed():
    reg = B.load_registry(VB.REGISTRY)
    assert set(reg["deferred"]) == {"mlaad_by_family", "asvspoof5_eval", "reverb_codec_telephony_shortcrop_cohorts",
                                    "sealed_set", "actDCF"}
    assert not set(reg["deferred"]) & set(reg["populations"])


@needs_data
def test_registry_hashes_match_disk():
    assert VB.check(B.load_registry(VB.REGISTRY), trees=True) == []


# ---------------------------------------------------------------------------
# S3. Metrics
# ---------------------------------------------------------------------------


def _brute(y, s, pi, cfa, cmiss):
    best = 1.0
    for t in np.r_[np.unique(s), np.inf]:
        pfa = np.mean(s[y == 0] >= t)
        pmiss = np.mean(s[y == 1] < t)
        best = min(best, (cfa * pfa * (1 - pi) + cmiss * pmiss * pi) / min(cfa * (1 - pi), cmiss * pi))
    return best


def test_scorecard_primary_is_package05():
    assert B.PRIMARY == "package05" and B.READINGS["package05"] == (0.5, 1.0, 4.0)
    cand = json.loads((REPO / "benchmarks/v2_60_candidates.json").read_text())
    assert cand["primary_reading"] == "package05"


def test_no_bare_min_cost_default_in_benchmark():
    tree = ast.parse((REPO / "src/hearsay/benchmark.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "min_cost"]
    assert calls, "benchmark.py must call min_cost"
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert {"c_fa", "c_miss"} <= kw and len(c.args) >= 3, ast.dump(c)


def test_package03_formula_hand_case():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.5, 0.9])
    # P_FA + (1.2/0.7)*P_miss; best at t=0.5: P_FA 1/3, P_miss 0
    assert B.reading(y, s, "package03") == pytest.approx(1 / 3, abs=1e-12)
    assert B.reading(y, s, "package03") == pytest.approx(_brute(y, s, 0.3, 1.0, 4.0), abs=1e-12)


def test_package05_equals_sponsor_flip():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    s = np.round(y + rng.normal(0, 1, 2000), 1)  # heavy ties
    assert B.reading(y, s, "package05") == pytest.approx(sponsor_min_dcf(y, s, flip=True), abs=1e-12)


def test_min_cost_with_heavy_ties():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 500)
    s = np.round(y * 0.5 + rng.normal(0, 1, 500))
    for name, (pi, cfa, cmiss) in B.READINGS.items():
        assert B.reading(y, s, name) == pytest.approx(_brute(y, s, pi, cfa, cmiss), abs=1e-12)
        assert B.counts_at(y, s, name)["cost"] == pytest.approx(B.reading(y, s, name), abs=1e-12)


def test_all_real_threshold_included():
    y = np.array([0] * 50 + [1] * 2)
    s = np.r_[np.linspace(0.5, 1.0, 50), [0.0, 0.1]]  # spoofs below every real: calling all real is best
    c = B.counts_at(y, s, "brief")
    assert c["threshold"] == "inf" and c["n_fa"] == 0 and c["n_miss"] == 2


def test_counts_carry_reading_and_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])
    for k in B.READINGS:
        c = B.counts_at(y, s, k)
        assert c["reading"] == k and c["threshold"] is not None and {"n_fa", "n_miss"} <= set(c)


def test_single_class_population_reports_na_not_crash():
    y = np.ones(10, dtype=int)
    s = np.linspace(0, 1, 10)
    r = B.readings(y, s)
    assert all(v is None for v in r.values())
    assert B.counts_at(y, s, "package05")["threshold"] is None
    ft = B.fold_table(np.r_[y, [0, 1]], np.r_[s, [0.2, 0.9]], np.array(["0"] * 10 + ["1", "1"]))
    assert ft["per_fold"]["0"]["package05"] is None and ft["macro"]["package05"] is not None


# ---------------------------------------------------------------------------
# S4. Rank and fusion primitives
# ---------------------------------------------------------------------------


def test_rank_left_side_and_n_train_denominator():
    ref = np.array([1.0, 2.0, 2.0, 3.0])
    assert np.array_equal(B.rank_vs(ref, np.array([2.0, 0.0, 4.0])), np.array([1 / 4, 0.0, 1.0]))
    logit = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fold = np.array(["0", "0", "1", "1", "2", "2"])
    r = B.nested_ranks(logit, fold)
    assert r[0] == 0.0 and r[5] == 4 / 4  # denominators are the 4 training rows


def test_blend_accumulates_in_weight_order():
    rng = np.random.default_rng(5)
    a, b, c = rng.random(1000), rng.random(1000), rng.random(1000)
    got = B.blend({"a": a, "b": b, "c": c}, {"a": 0.6, "b": 0.2, "c": 0.2})
    assert np.array_equal(got, ((0.6 * a) + (0.2 * b)) + (0.2 * c))
    got0 = B.blend({"a": a, "b": b}, {"a": 1.0, "b": 0.0, "z": 0.0})
    assert np.array_equal(got0, a)


def test_e_step_boundaries():
    base = np.array([0.5, 0.5000001, 0.9, 0.9, 0.9])
    m3 = np.array([-5.0, -5.0, -3.0, -3.0000001, -2.0])
    out, mask = B.apply_e(base, m3)
    assert mask.tolist() == [False, True, False, True, False]
    assert out[1] == base[1] * 0.5 and out[3] == 0.45 and out[0] == 0.5


def test_e_uses_base_before_platt(world):
    bench = bench_with(world)
    rule = B.Rule("SHIP", dict(B.SHIP_WEIGHTS))
    r = bench.score_rule(rule, "inner")
    m3 = bench.cols[B.E_COLUMN]["inner"]
    assert np.array_equal(r["e_mask"], (m3 < -3.0) & (r["base"] > 0.5))
    assert np.array_equal(r["fused"], np.where(r["e_mask"], r["base"] * 0.5, r["base"]))


def test_e_nan_input_raises():
    with pytest.raises(B.BenchmarkError, match="non-finite"):
        B.apply_e(np.array([0.9, 0.2]), np.array([np.nan, -4.0]))
    with pytest.raises(B.BenchmarkError, match="non-finite"):
        B.apply_e(np.array([np.nan, 0.2]), np.array([-4.0, -4.0]))


def test_platt_monotone_metrics_unchanged():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 3000)
    fused = np.clip(y * 0.3 + rng.random(3000) * 0.7, 0, 1)
    platt = {"a": 17.69, "b": -8.15, "prior_shift": -0.847}
    assert B.readings(y, fused) == B.readings(y, B.shipped_probability(fused, platt))


def test_rule_weights_validated():
    for bad in ({"a": -0.1, "b": 1.1}, {"a": np.nan, "b": 1.0}, {"a": 0.5, "b": 0.6},
                {"a": 0.5, "spectra_aasist": 0.5}, {}):
        with pytest.raises(B.BenchmarkError):
            B.validate_weights(bad)
    with pytest.raises(B.BenchmarkError, match="unknown"):
        B.validate_weights({"a": 1.0}, known={"b"})
    B.validate_weights({"a": 0.6, "b": 0.4})


# ---------------------------------------------------------------------------
# S5. Readouts
# ---------------------------------------------------------------------------


def test_readout_has_pooled_perfold_macro_f0_f4(world):
    bench = bench_with(world)
    card = VB.rule_scorecard(bench, B.Rule("SHIP", dict(B.SHIP_WEIGHTS)), with_holdout=False)
    inner = card["populations"]["inner"]
    assert {"pooled", "per_fold", "macro", "counts"} <= set(inner)
    assert set(inner["per_fold"]) == {"0", "1", "2", "3", "4"}
    assert set(inner["pooled"]) == {"brief", "package05", "package03", "eer"}


def test_readout_has_with_and_without_e(world):
    bench = bench_with(world)
    card = VB.rule_scorecard(bench, B.Rule("SHIP", dict(B.SHIP_WEIGHTS)), with_holdout=False)
    assert "without_e" in card and "pooled" in card["without_e"]["inner"]
    assert "e_fires_inner" in card and set(card["e_fires_inner"]["per_fold_real"]) == {"0", "1", "2", "3", "4"}


def test_dominance_spearman_on_real_rows():
    rng = np.random.default_rng(7)
    a, b = rng.random(500), rng.random(500)
    d = B.dominance({"a": a, "b": b}, 0.9 * a + 0.1 * b, {"a": 0.3, "b": 0.7})
    assert d["a"]["flag"] and not d["b"]["flag"]
    assert d["a"]["weight_share"] == 0.3


def test_macro_round_then_mean_matches_research():
    y = np.array([0, 1] * 30)
    s = np.random.default_rng(8).random(60)
    fold = np.array([str(i % 3) for i in range(60)])
    ft = B.fold_table(y, s, fold)
    for k in B.READINGS:
        vals = [round(B.reading(y[fold == f], s[fold == f], k), 4) for f in ("0", "1", "2")]
        assert ft["macro"][k] == round(float(np.mean(vals)), 4)


def test_all_candidates_scored_on_identical_rows(world):
    bench = bench_with(world)
    r1 = bench.score_rule(B.Rule("a", dict(B.SHIP_WEIGHTS)), "inner")
    r2 = bench.score_rule(B.Rule("b", {"m1b_v3": 0.5, "m5_xlsr_ft": 0.5}), "inner")
    assert len(r1["fused"]) == len(r2["fused"]) == bench.pops["inner"].n


def test_option_b_candidates_report_not_scored_where_columns_missing(world):
    bench = bench_with(world, splits=("inner", "holdout", "itw"))
    root, pops, rng = world
    bench.add(read(write_scores(root, pops, "wavlm_l", rng, splits=("inner", "holdout")), root, name="wavlm_l"))
    r = bench.score_rule(B.Rule("C1", {"m1b_v3": 0.5, "wavlm_l": 0.5}), "itw")
    assert r == {"scored": False, "reason": "no scores for ['wavlm_l'] on itw"}


# ---------------------------------------------------------------------------
# S6. Bootstrap
# ---------------------------------------------------------------------------


def test_lj_is_one_cluster(world):
    _, pops, _ = world
    inner = pops["inner"]
    lj = inner.stratum == "lj"
    assert lj.any() and set(inner.cluster[lj]) == {"lj"}


def test_lj_stratum_always_present(world):
    _, pops, _ = world
    inner = pops["inner"]
    lj_idx = set(np.flatnonzero(inner.stratum == "lj"))
    for ix in B.bootstrap_draws(inner.cluster, inner.stratum, n=50, seed=0, lj_mode="fixed"):
        vals, counts = np.unique(ix, return_counts=True)
        got = {int(v): int(c) for v, c in zip(vals, counts) if v in lj_idx}
        assert set(got) == lj_idx and set(got.values()) == {1}


def test_inner_bootstrap_clusters(world):
    _, pops, _ = world
    inner = pops["inner"]
    _fixed, groups = B.bootstrap_strata(inner.cluster, inner.stratum, "fixed")
    assert set(groups) == {"libri", "spoof"}
    for ix in B.bootstrap_draws(inner.cluster, inner.stratum, n=20, seed=1):
        spoof = ix[inner.y[ix] == 1]
        # whole clusters only: each drawn generator contributes all of its rows
        for g in set(inner.cluster[spoof]):
            n_rows = int((inner.cluster == g).sum())
            assert int((inner.cluster[spoof] == g).sum()) % n_rows == 0
        assert len(set(inner.cluster[ix][inner.stratum[ix] == "spoof"])) <= len(groups["spoof"])


def test_bootstrap_seeded_and_paired(world):
    _, pops, rng = world
    inner = pops["inner"]
    a = inner.y + rng.normal(0, 1, inner.n)
    b = inner.y * 1.5 + rng.normal(0, 1, inner.n)
    r1 = B.cluster_bootstrap(inner.y, a, b, inner.cluster, inner.stratum, n=60, seed=2)
    r2 = B.cluster_bootstrap(inner.y, a, b, inner.cluster, inner.stratum, n=60, seed=2)
    assert r1 == r2
    same = B.cluster_bootstrap(inner.y, a, a, inner.cluster, inner.stratum, n=30, seed=2)
    assert all(same[k]["p5"] == same[k]["p95"] == 0.0 for k in B.READINGS)


def test_bootstrap_all_skipped_raises():
    y = np.zeros(30, dtype=int)
    cl = np.array([f"c{i % 6}" for i in range(30)])
    st = np.array(["libri"] * 30)
    with pytest.raises(B.BenchmarkError, match="single-class"):
        B.cluster_bootstrap(y, np.random.default_rng(0).random(30), np.random.default_rng(1).random(30), cl, st, n=10)


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(B.BenchmarkError, match="length"):
        B.cluster_bootstrap(np.array([0, 1]), np.array([0.1, 0.2]), np.array([0.1]), np.array(["a", "b"]),
                            np.array(["libri", "spoof"]))


def test_k7_sensitivity_row_and_degenerate_folds_marked(world):
    _, pops, rng = world
    inner = pops["inner"]
    a = inner.y + rng.normal(0, 1, inner.n)
    sens = B.cluster_bootstrap(inner.y, a, a + 0.1, inner.cluster, inner.stratum, n=20, lj_mode="cluster")
    assert sens["lj_mode"] == "cluster" and not sens["degenerate"]
    f3 = inner.fold == "3"
    d = B.cluster_bootstrap(inner.y[f3], a[f3], a[f3], inner.cluster[f3], inner.stratum[f3], n=20)
    assert d["degenerate"] and "spoof" in d["reason"]


# ---------------------------------------------------------------------------
# Ledger details (K8)
# ---------------------------------------------------------------------------


PINNED = {rel: {"sha256": f"sha-{c}"} for c, rel in B.REFERENCE_COLUMN_FILES.items()}
PINNED_SHAS = {c: f"sha-{c}" for c in B.REFERENCE_COLUMN_FILES}


def test_reference_rules_exempt_from_ledger(tmp_path):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    for _ in range(2):
        r = B.read_holdout([B.column_def("m1b_v3", "sha-m1b_v3")], lambda: {"ok": 1}, ledger=led, holdout_sha="h",
                           pinned=PINNED)
        assert r["ledger"].startswith("reference")
    assert led.count() == 0


def test_reference_exemption_limited_to_spec_refs():
    ship = B.ship_def(PINNED_SHAS)
    assert B.is_reference([ship], PINNED) and B.is_reference([B.column_def("m1b_v3", "sha-m1b_v3")], PINNED)
    assert not B.is_reference([B.column_def("m5_xlsr_ft", "sha-m5_xlsr_ft")], PINNED)
    assert not B.is_reference([{**ship, "weights": {"m1b_v3": 0.5, "handcrafted_v5": 0.25, "m5_xlsr_ft": 0.25}}],
                              PINNED)
    assert not B.is_reference([ship, B.column_def("wavlm_l", "x")], PINNED)
    assert not B.is_reference([], PINNED)
    assert not B.is_reference([B.column_def("m1b_v3", "sha-m1b_v3")], None)  # no pins, no exemption


def test_reference_exemption_bound_to_pinned_file(tmp_path):
    """A different file that calls itself m1b_v3 is not the reference: it is ledgered."""
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    imposter = B.column_def("m1b_v3", "sha-of-some-other-file")
    assert not B.is_reference([imposter], PINNED)
    n = []
    B.read_holdout([imposter], lambda: n.append(1) or {"v": 1}, ledger=led, holdout_sha="h", pinned=PINNED)
    B.read_holdout([imposter], lambda: n.append(1) or {"v": 2}, ledger=led, holdout_sha="h", pinned=PINNED)
    assert len(n) == 1 and led.count() == 1
    ship_other = B.ship_def({**PINNED_SHAS, "m5_xlsr_ft": "changed"})
    assert not B.is_reference([ship_other], PINNED)


def test_ledger_refuses_when_authority_lost_but_audit_has_key(tmp_path):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    defs = [{"kind": "fixed", "weights": {"x": 1.0}, "e": True}]
    B.read_holdout(defs, lambda: {"v": 1}, ledger=led, holdout_sha="h")
    for f in (tmp_path / "a").glob("*.json"):
        f.unlink()  # the gitignored authority dir is lost; the tracked audit copy remains
    with pytest.raises(B.HoldoutAlreadyRead, match="audit log"):
        B.read_holdout(defs, lambda: {"v": 2}, ledger=led, holdout_sha="h")


def test_ledger_rebuild_and_alias_not_counted(tmp_path):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl", tmp_path / "m.jsonl"])
    old = B.ledger_key([{"kind": "column", "name": "hc"}], "h")
    B.read_holdout([{"kind": "column", "name": "hc"}], lambda: {"v": 1}, ledger=led, holdout_sha="h")
    new = B.ledger_key([B.column_def("hc", "s")], "h")
    led.alias(new, old, {"code_sha": "c"})
    assert led.count() == 1 and led.persisted(new)["readout"] == {"v": 1}
    r = B.read_holdout([B.column_def("hc", "s")], lambda: pytest.fail("aliased read must not recompute"),
                       ledger=led, holdout_sha="h")
    assert r["readout"] == {"v": 1}
    (tmp_path / "m.jsonl").unlink()
    assert led.rebuild() == 3  # reserve + complete for the read, alias for the alias
    events = [json.loads(x)["event"] for x in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert sorted(events) == ["alias", "complete", "reserve"]
    assert led.rebuild() == 0


def test_code_sha_sees_dirty_tree(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    GIT = "git"
    subprocess.run([GIT, "init", "-q"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([GIT, "add", "a.py"], cwd=repo, check=True)
    subprocess.run([GIT, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "a"], cwd=repo, check=True)
    clean_sha = B.code_sha(repo)
    assert "+dirty" not in clean_sha
    (repo / "a.py").write_text("x = 2\n")
    assert B.code_sha(repo).startswith(clean_sha + "+dirty:")


def test_ledger_written_before_holdout_metrics(tmp_path):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    defs = [{"kind": "fixed", "weights": {"x": 1.0}, "e": False}]
    key = B.ledger_key(defs, "h")
    seen = {}

    def compute():
        seen["exists"] = (tmp_path / "a" / f"{key}.json").exists()
        seen["status"] = json.loads((tmp_path / "a" / f"{key}.json").read_text())["status"]
        return {"v": 1}

    B.read_holdout(defs, compute, ledger=led, holdout_sha="h")
    assert seen == {"exists": True, "status": "reserved"}
    assert led.persisted(key)["readout"] == {"v": 1}


def test_tracked_ledger_unchanged_by_tests():
    """The autouse fixture points every test at a tmp ledger."""
    led = B.HoldoutLedger.default()
    assert not str(led.dir).startswith(str(REPO / "outputs"))
    assert all(not str(p).startswith(str(REPO / "benchmarks")) for p in led.jsonl)


def test_ledger_env_override_only_under_pytest(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARSAY_HOLDOUT_LEDGER", str(tmp_path / "x"))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    led = B.HoldoutLedger.default()
    assert led.dir == B.REPO / "outputs/v2/benchmark/holdout_ledger"


def test_ledger_concurrent_reservation_one_wins(tmp_path):
    code = (
        "import sys, json; from hearsay import benchmark as B; "
        "led = B.HoldoutLedger(sys.argv[1], [sys.argv[1] + '/t.jsonl']); "
        "print(led.reserve('k' * 64, {'code_sha': sys.argv[2], 'input_shas': {}})['status'])"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "a"), f"c{i}"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, cwd=REPO) for i in range(4)]
    outs = [p.communicate() for p in procs]
    statuses = [o[0].strip() for o in outs]
    assert statuses.count("new") == 1
    assert sum("HoldoutAlreadyRead" in o[1] for o in outs) == 3


def test_ledger_version_bump_cannot_mint_new_read(tmp_path, monkeypatch):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    defs = [{"kind": "fixed", "weights": {"x": 1.0}, "e": True}]
    n = []
    B.read_holdout(defs, lambda: n.append(1) or {"v": 1}, ledger=led, holdout_sha="h")
    monkeypatch.setattr(B, "BENCHMARK_VERSION", "v1.1")
    B.read_holdout(defs, lambda: n.append(1) or {"v": 2}, ledger=led, holdout_sha="h")
    assert len(n) == 1


def test_ledger_regeneration_reuses_persisted_readout(tmp_path):
    led = B.HoldoutLedger(tmp_path / "a", [tmp_path / "t.jsonl"])
    defs = [{"kind": "column", "name": "hc"}]
    B.read_holdout(defs, lambda: {"pooled": {"brief": 0.1}}, ledger=led, holdout_sha="h")
    r = B.read_holdout(defs, lambda: pytest.fail("must not recompute"), ledger=led, holdout_sha="h")
    assert r["readout"] == {"pooled": {"brief": 0.1}}
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert [json.loads(x)["event"] for x in lines] == ["reserve", "complete"]


# ---------------------------------------------------------------------------
# Script behaviour
# ---------------------------------------------------------------------------


def test_score_name_collision_refused(tmp_path):
    p = tmp_path / "x.json"
    VB.write_json(p, {"a": 1}, force=False)
    with pytest.raises(SystemExit, match="refusing"):
        VB.write_json(p, {"a": 2}, force=False)
    VB.write_json(p, {"a": 2}, force=True)
    assert json.loads(p.read_text()) == {"a": 2}


def test_import_hygiene_benchmark():
    code = "import sys, hearsay.benchmark; print('torch' in sys.modules, 'lightgbm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True)
    assert out.stdout.strip() == "False False"


def test_l20_adapter_rows_json_order_and_basename_prefix(world, tmp_path):
    root, pops, rng = world
    inner_keys = list(pops["inner"].keys)
    extra = [f"data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_{i}.flac" for i in range(7)]
    order = rng.permutation(len(inner_keys) + 7)
    allp = np.array(inner_keys + extra)[order]
    fold_map = dict(zip(pops["inner"].keys, pops["inner"].fold))
    folds = [int(fold_map.get(p, "0")) for p in allp]
    oof = rng.normal(size=len(allp))
    rows = {"inner_path": [f"{root}/{p}" for p in allp], "inner_fold": folds,
            "holdout_path": list(pops["holdout"].keys), "test_name": [Path(k).name for k in pops["test"].keys],
            "itw_path": list(pops["itw"].keys)}
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    np.savez(tmp_path / "L.npz", oof=oof, holdout=rng.normal(size=pops["holdout"].n),
             test=rng.normal(size=pops["test"].n), itw=rng.normal(size=pops["itw"].n))
    orig = B._MAIN
    try:
        B._MAIN = root
        df = VB.adapt_l20(tmp_path / "L.npz", tmp_path / "rows.json", tmp_path / "out.csv", pops=pops)
    finally:
        B._MAIN = orig
    inner = df[df["split"] == "inner_oof"].set_index("path")["logit"]
    pos = {p: i for i, p in enumerate(allp)}
    assert len(inner) == pops["inner"].n
    for k in inner_keys[:25]:
        assert inner[k] == oof[pos[k]]
    test = df[df["split"] == "test"]["path"].tolist()
    assert test == [f"data/nsa/HackGTHearsayTesting/{Path(k).name}" for k in pops["test"].keys]


def test_cohort_reader_ignores_precomputed_fused(tmp_path):
    rows = []
    for kind in B.COHORT_KINDS:
        for i in range(4):
            rows.append({"key": f"k{i}|{kind}", "path": f"data/nsa/c{i}.wav", "kind": kind, "label": "spoof",
                         "m1b_v3": float(i), "fused": 999.0, "fused_base": 999.0, "e_applied": 1.0})
    p = tmp_path / "w.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    out = B.read_cohort_file(p, column="m1b_v3", main=tmp_path, repo=tmp_path)
    assert set(out.columns) == {"key", "kind", "logit"} and 999.0 not in out["logit"].to_numpy()
    with pytest.raises(B.BenchmarkError, match="partial cohort"):
        pd.DataFrame(rows[:-1]).to_csv(p, index=False)
        B.read_cohort_file(p, column="m1b_v3", main=tmp_path, repo=tmp_path)


def test_cohort_deltas_vs_own_none_kind(world):
    bench = bench_with(world)
    _root, pops, rng = world
    keys = pops["holdout"].keys
    frames = {}
    for c in ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist"):
        rows = []
        for j, kind in enumerate(B.COHORT_KINDS):
            for k, y in zip(keys, pops["holdout"].y):
                rows.append({"key": k, "kind": kind, "logit": float(2 * y - 1 - 0.3 * j + rng.normal())})
        frames[c] = pd.DataFrame(rows)
    labels = pd.Series(pops["holdout"].y, index=keys)
    out = VB.cohort_readout(bench, {"SHIP": B.Rule("SHIP", dict(B.SHIP_WEIGHTS))}, frames, labels)
    for kind in B.COHORT_KINDS:
        for k in B.READINGS:
            assert out["SHIP"][kind]["delta_vs_none"][k] == pytest.approx(out["SHIP"][kind][k] - out["SHIP"]["none"][k])
    assert out["SHIP"]["none"]["delta_vs_none"]["brief"] == 0.0


# ---------------------------------------------------------------------------
# S7. Reference reproduction (needs data)
# ---------------------------------------------------------------------------

REFS = {
    ("m1b", "inner"): (0.3013, 0.3307, 0.2280),
    ("m1b", "holdout"): (0.0717, 0.0571, None),
    ("m1b", "itw"): (0.342, 0.2955, None),
    ("ship_t", "inner"): (0.1351, 0.2997, None),
    ("ship_t", "holdout"): (0.0065, 0.0087, None),
    ("ship_t", "itw"): (0.228, 0.2385, None),
    ("ship_n", "inner"): (0.1558, 0.3390, None),
}


@pytest.fixture(scope="module")
def real_bench():
    if not HAVE_DATA:
        pytest.skip("needs the gitignored score exports")
    return VB.load_bench(["m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist"])


def _check(r, ref):
    for k, v in zip(("brief", "package05", "package03"), ref):
        if v is not None:
            assert abs(r[k] - v) <= 5e-5 + 1e-12, (k, r[k], v)


def _ship(real_bench, pop, refs):
    return real_bench.score_rule(B.Rule("SHIP", dict(B.SHIP_WEIGHTS)), pop, refs)["fused"]


@needs_data
def test_ref_m1b_inner(real_bench):
    _check(B.readings(real_bench.pops["inner"].y, real_bench.cols["m1b_v3"]["inner"]), REFS[("m1b", "inner")])


@needs_data
def test_ref_m1b_holdout(real_bench):
    _check(B.readings(real_bench.pops["holdout"].y, real_bench.cols["m1b_v3"]["holdout"]), REFS[("m1b", "holdout")])


@needs_data
def test_ref_m1b_itw(real_bench):
    _check(B.readings(real_bench.pops["itw"].y, real_bench.cols["m1b_v3"]["itw"]), REFS[("m1b", "itw")])


@needs_data
def test_ref_ship_transductive_inner(real_bench):
    _check(B.readings(real_bench.pops["inner"].y, _ship(real_bench, "inner", "transductive")), REFS[("ship_t", "inner")])


@needs_data
def test_ref_ship_transductive_holdout(real_bench):
    _check(B.readings(real_bench.pops["holdout"].y, _ship(real_bench, "holdout", "nested")), REFS[("ship_t", "holdout")])


@needs_data
def test_ref_ship_transductive_itw(real_bench):
    _check(B.readings(real_bench.pops["itw"].y, _ship(real_bench, "itw", "nested")), REFS[("ship_t", "itw")])


@needs_data
def test_ref_ship_nested_inner(real_bench):
    _check(B.readings(real_bench.pops["inner"].y, _ship(real_bench, "inner", "nested")), REFS[("ship_n", "inner")])


@needs_data
def test_constants_rank_refs_equal_sorted_inner_logits_exact(real_bench):
    c = json.loads((REPO / "models/fusion_v2/constants.json").read_text())
    for col, ref in c["rank_ref_inner_oof_sorted"].items():
        assert np.array_equal(np.array(ref), np.sort(real_bench.cols[col]["inner"])), col


@needs_data
def test_shipped_tsv_resolved_from_main_checkout():
    p = B.shipped_tsv_path()
    assert p.exists() and p.parent.name == "submissions"
    assert p.parent.parent == B.main_root()


@needs_data
def test_c0_reproduces_shipped_tsv_on_test(real_bench):
    c = json.loads((REPO / "models/fusion_v2/constants.json").read_text())
    fused = _ship(real_bench, "test", "nested")
    p = B.shipped_probability(fused, c["platt"])
    tsv = pd.read_csv(B.shipped_tsv_path(), sep="\t").set_index("filename")["cm-score"]
    want = tsv.reindex([Path(k).name for k in real_bench.pops["test"].keys]).to_numpy()
    assert np.max(np.abs(p - want)) <= 1e-12


@needs_data
def test_rerun_bit_identical():
    code = (
        "import json; from hearsay import benchmark as B; import importlib.util as u; "
        "s=u.spec_from_file_location('vb','scripts/v2_benchmark.py'); m=u.module_from_spec(s); s.loader.exec_module(m); "
        "b=m.load_bench(['m1b_v3','handcrafted_v5','m5_xlsr_ft','spectra_aasist']); "
        "c=m.rule_scorecard(b, B.Rule('SHIP', dict(B.SHIP_WEIGHTS)), with_holdout=False, boot_n=20); "
        "c['populations'].pop('test', None); assert 'inner' in c['populations']; "
        "r=B.cluster_bootstrap(b.pops['inner'].y, b.score_rule(B.Rule('S',dict(B.SHIP_WEIGHTS)),'inner')['fused'], "
        "b.score_rule(B.Rule('E',{'m1b_v3':0.5,'m5_xlsr_ft':0.5}),'inner')['fused'], b.pops['inner'].cluster, "
        "b.pops['inner'].stratum, n=20); print(B.dumps({'card': c, 'boot': r}))"
    )
    outs = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=env,
                                   check=True).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


# ---------------------------------------------------------------------------
# Added after the pre-audit critique
# ---------------------------------------------------------------------------


def test_dominance_check_flags_column_over_share():
    """L: a column that drives the fused score far beyond its weight (the way a
    free fit hands Spectra the score) is flagged; a fairly weighted one is not."""
    rng = np.random.default_rng(21)
    driver, other = rng.random(800), rng.random(800)
    fused = 0.98 * driver + 0.02 * other
    d = B.dominance({"spectra_like": driver, "other": other}, fused, {"spectra_like": 0.1, "other": 0.9})
    assert d["spectra_like"]["flag"] is True and d["other"]["flag"] is False
    d2 = B.dominance({"m1b": driver}, fused, {"m1b": 0.6})
    assert d2["m1b"]["flag"] is False


def test_bootstrap_ci_marks_holdout_degenerate_and_counts_missing_lj(world):
    _, pops, rng = world
    h = pops["holdout"]
    ci = B.bootstrap_ci(h.y, h.y + rng.normal(0, 1, h.n), h.cluster, h.stratum, n=10)
    assert ci["degenerate"] and "spoof" in ci["reason"]
    inner = pops["inner"]
    s = inner.y + rng.normal(0, 1, inner.n)
    fixed = B.bootstrap_ci(inner.y, s, inner.cluster, inner.stratum, n=200, lj_mode="fixed")
    sens = B.bootstrap_ci(inner.y, s, inner.cluster, inner.stratum, n=200, lj_mode="cluster")
    assert fixed["draws_missing_lj"] == 0 and sens["draws_missing_lj"] > 0
    for k in B.READINGS:
        assert fixed[k]["p5"] <= fixed[k]["p50"] <= fixed[k]["p95"]


def test_scorecards_carry_bootstrap_intervals(world):
    bench = bench_with(world)
    card = VB.rule_scorecard(bench, B.Rule("SHIP", dict(B.SHIP_WEIGHTS)), with_holdout=False, boot_n=30)
    assert {"lj_fixed", "lj_resampled"} <= set(card["populations"]["inner"]["bootstrap"])
    assert "stratified" in card["populations"]["itw"]["bootstrap"]
    col = VB.column_scorecard(bench, "m1b_v3", with_holdout=False, boot_n=30)
    assert col["populations"]["inner"]["bootstrap"]["lj_fixed"]["n"] == 30


def test_commands_refuse_on_registry_mismatch(monkeypatch):
    if not VB.REGISTRY.exists():
        pytest.skip("no registry")
    monkeypatch.setattr(VB, "check", lambda reg, trees=True: ["sha256 mismatch outputs/detector_scores/m1b_v3.csv"])
    with pytest.raises(SystemExit, match="registry check failed"):
        VB.require_registry()
    ns = type("A", (), {"name": "x", "file": "f.csv", "itw": None, "holdout": False, "force": False, "lane": "t",
                        "boot_n": 0})()
    with pytest.raises(SystemExit, match="registry check failed"):
        VB.cmd_score(ns)


def test_score_name_sanitized():
    assert VB.safe_name("wavlm_l.v2-hn") == "wavlm_l.v2-hn"
    for bad in ("../evil", "a/b", "", ".hidden", "x..y"):
        with pytest.raises(SystemExit):
            VB.safe_name(bad)
