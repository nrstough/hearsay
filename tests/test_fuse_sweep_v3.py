"""Post-draft gate sweep (docs/specs/2026-09-26_post-draft-gate-sweep.md): the pure functions of
scripts/fuse_sweep_v3.py, every gate rule's counterfactual, the diagnostics, the W4 bake-off, the
write guards, plus needs_data self-checks against the shipped numbers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay.metrics import cost_at, min_cost

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fuse_sweep_v3", REPO / "scripts" / "fuse_sweep_v3.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

HAVE_DATA = (REPO / "outputs/detector_scores/m1b_v3.csv").exists() and v3.SHIPPED_TSV.exists()


def _cells(**kw):
    d = {c: 0.0 for c in v3.CELLS}
    d.update(kw)
    return d


def _pass_d():
    # every rule satisfied: ITW +0.03 / +0.001, inner +0.02 / 0, holdout 0 / +0.001 (averse improves on 2 of 3)
    return _cells(itw_brief=0.03, itw_averse=0.001, inner_brief=0.02, inner_averse=0.0, holdout_brief=0.0,
                  holdout_averse=0.001)


# ------------------------------------------------------------------ costs (the named failure mode)
def test_brief_is_min_cost_pi03():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 400), rng.normal(size=400)
    assert v3.brief(y, s) == min_cost(y, s, 0.3, c_fa=4.0, c_miss=1.0)


def test_averse_is_min_cost_sponsor():
    rng = np.random.default_rng(2)
    y, s = rng.integers(0, 2, 400), rng.normal(size=400)
    assert v3.averse(y, s) == min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0)


def test_brief_weights_recover_9_33_to_1():
    # 100 real, 100 spoof; threshold 0.5. One FA alone vs one miss alone.
    y = np.r_[np.zeros(100), np.ones(100)]
    s_fa = np.r_[np.zeros(99), 1.0, np.ones(100)]
    s_miss = np.r_[np.zeros(100), 0.0, np.ones(99)]
    fa, miss = cost_at(y, s_fa, 0.5, 0.3), cost_at(y, s_miss, 0.5, 0.3)
    assert fa / miss == pytest.approx(4 * 0.7 / 0.3)  # 9.33 : 1


def test_averse_weights_recover_1_to_4():
    y = np.r_[np.zeros(100), np.ones(100)]
    s_fa = np.r_[np.zeros(99), 1.0, np.ones(100)]
    s_miss = np.r_[np.zeros(100), 0.0, np.ones(99)]
    fa = cost_at(y, s_fa, 0.5, 0.5, c_fa=1.0, c_miss=4.0)
    miss = cost_at(y, s_miss, 0.5, 0.5, c_fa=1.0, c_miss=4.0)
    assert miss / fa == pytest.approx(4.0)


def test_brief_differs_from_averse_on_asymmetric():
    # one real scored above every spoof: brief pays it 9.33x, averse cheaply
    y = np.r_[np.zeros(50), np.ones(50)]
    s = np.r_[np.linspace(0, 0.4, 49), 2.0, np.linspace(0.6, 1.0, 50)]
    assert v3.brief(y, s) != v3.averse(y, s)


# ------------------------------------------------------------------ fusion arithmetic
def test_rank_vs_matches_hand():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(v3.rank_vs(ref, np.array([0.0, 2.0, 2.5, 5.0])), [0.0, 0.25, 0.5, 1.0])


def test_blend_order_and_weights():
    r = {"a": np.array([1.0]), "b": np.array([0.5])}
    assert v3.blend(r, {"a": 0.6, "b": 0.4})[0] == pytest.approx(0.8)


def test_manifest_weights_sum_to_one():
    assert v3.CURRENT_W == {"m1b_v3": 0.6, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2}
    assert v3.CANDIDATES["W4"]["weights"] == {"m1b_v3": 0.4, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.4}
    assert v3.CANDIDATES["P_wl"]["weights"] == {"m1b_v3": 0.4, "wavlm_l": 0.2, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2}
    for c in v3.CANDIDATES.values():
        assert sum(c["weights"].values()) == pytest.approx(1.0)
    assert list(v3.CANDIDATES) == ["T2", "W4", "P_wl", "H_noise"]


@pytest.mark.parametrize("m3,base,want", [(-7.0, 0.8, 0.2), (-4.0, 0.8, 0.4), (-7.0, 0.4, 0.4), (-2.0, 0.8, 0.8),
                                          (-3.0, 0.8, 0.8), (-6.0, 0.8, 0.4)])  # fmt: skip
def test_apply_tiers_t2_point_cases(m3, base, want):
    assert v3.apply_tiers(np.array([base]), np.array([m3]), v3.T2_TIERS)[0] == pytest.approx(want)


def test_apply_tiers_rejects_misordered_tiers():
    with pytest.raises(AssertionError):
        v3.apply_tiers(np.array([0.8]), np.array([-7.0]), ((-6.0, 0.25), (-3.0, 0.5)))


def test_e_tiers_equal_shipped_e_step():
    rng = np.random.default_rng(3)
    base, m3 = rng.uniform(size=1000), rng.normal(-3, 3, 1000)
    want = np.where((m3 < -3) & (base > 0.5), base * 0.5, base)
    assert np.array_equal(v3.apply_tiers(base, m3, v3.E_TIERS), want)


def test_inner_threshold_includes_all_real():
    y = np.r_[np.zeros(10), np.ones(10)]
    assert v3.inner_threshold(y, np.zeros(20)) == np.inf  # constant column: calling all real is optimal


# ------------------------------------------------------------------ the gate
def test_gate_all_pass():
    g = v3.gate(_pass_d(), 0.99, True)
    assert g["verdict"] == "PASS" and not g["failed"] and not g["doubled"]


@pytest.mark.parametrize("rule,change", [
    ("2a_itw_brief_gain", {"itw_brief": 0.0199}),
    ("2b_itw_averse_reg", {"itw_averse": -0.0021}),
    ("3a_inner_brief_gain", {"inner_brief": 0.0099}),
    ("3b_inner_averse_no_reg", {"inner_averse": -0.0001}),
    ("4a_holdout_brief", {"holdout_brief": -0.0101}),
    ("4b_holdout_averse", {"holdout_averse": -0.0101}),
    ("5_brief_2of3", {"inner_brief": 0.0}),  # brief improves on ITW only (3a also fails: 5 is implied by 2a+3a)
])
def test_gate_each_rule_fails(rule, change):
    d = _pass_d()
    d.update(change)
    g = v3.gate(d, 0.99, True)
    assert rule in g["failed"] and g["verdict"] == "FAIL"


def test_gate_rule5_averse_counts_populations():
    d = _pass_d()
    d.update(holdout_averse=0.0)  # averse improves on ITW only
    g = v3.gate(d, 0.99, True)
    assert g["rules"]["5_averse_2of3"] is False
    d.update(holdout_averse=0.0001)
    assert v3.gate(d, 0.99, True)["rules"]["5_averse_2of3"] is True


def test_gate_diagnostic_rule():
    g = v3.gate(_pass_d(), 0.99, False)
    assert g["failed"] == ["5_diagnostic"] and g["verdict"] == "FAIL"


def test_gate_boundaries():
    d = _pass_d()
    d.update(itw_brief=0.020, itw_averse=-0.002, holdout_brief=-0.010, holdout_averse=-0.010, inner_brief=0.010)
    r = v3.gate(d, 0.99, True)["rules"]
    assert r["2a_itw_brief_gain"] and r["2b_itw_averse_reg"] and r["4a_holdout_brief"] and r["4b_holdout_averse"]
    assert r["3a_inner_brief_gain"]


def test_deltas_round_before_compare():
    cur = _cells(itw_brief=0.22804, itw_averse=0.2)
    cand = _cells(itw_brief=0.20796, itw_averse=0.2)
    assert v3.deltas(cur, cand)["itw_brief"] == 0.0200  # 0.2280 - 0.2080 after rounding


def test_gate_spearman_doubles():
    d = _pass_d()
    d.update(itw_brief=0.03, inner_brief=0.015)
    g = v3.gate(d, 0.9739, True)
    assert g["doubled"] and {"2a_itw_brief_gain", "3a_inner_brief_gain"} <= set(g["failed"])
    assert "2b_itw_averse_reg" not in g["failed"] and "4a_holdout_brief" not in g["failed"]
    assert not v3.gate(d, 0.974, True)["doubled"]


def test_gate_nan_spearman_raises():
    with pytest.raises(ValueError):
        v3.gate(_pass_d(), float("nan"), True)


def test_room():
    d = _pass_d()
    d.update(itw_brief=0.03, itw_averse=0.03)
    assert v3.room(d, 19)["verdict"] == "PASS"
    assert v3.room(d, 18)["failed"] == ["catches_19"]
    d.update(itw_averse=0.0299)
    assert "itw_averse_gain_0.030" in v3.room(d, 19)["failed"]


@pytest.mark.parametrize("change,ok", [({}, True), ({"itw_brief": 0.0099}, False), ({"itw_averse": -0.0101}, False),
                                       ({"holdout_brief": -0.0501}, False), ({"inner_brief": -0.0301}, False)])
def test_standing_rule_cases(change, ok):
    d = _cells(itw_brief=0.01, itw_averse=-0.01, holdout_brief=-0.05, inner_brief=-0.03)
    d.update(change)
    assert v3.standing_rule(d)["ok"] is ok


def test_pareto_pick():
    a, b = _cells(itw_brief=0.1), _cells(itw_brief=0.2)
    assert v3.pareto_pick({}) is None
    assert v3.pareto_pick({"A": a}) == "A"
    assert v3.pareto_pick({"A": a, "B": b}) == "A"
    assert v3.pareto_pick({"A": a, "B": dict(a)}) is None  # all equal: nobody strictly better
    c = _cells(itw_brief=0.1, inner_brief=0.2)
    d = _cells(itw_brief=0.2, inner_brief=0.1)
    assert v3.pareto_pick({"C": c, "D": d}) is None


# ------------------------------------------------------------------ diagnostics
def test_t2_fires_counts_only_spoof_above_half():
    y = np.array([1, 1, 0, 1])
    base = np.array([0.8, 0.4, 0.9, 0.8])
    m3 = np.array([-7.0, -7.0, -7.0, -5.0])
    assert v3.t2_fires(y, base, m3) == {"spoof": 1, "bonafide": 1}


def test_new_catches_counts_current_misses_at_its_threshold():
    y = np.r_[np.ones(20), np.zeros(5)]
    cur = np.r_[np.zeros(20), np.zeros(5)]  # every spoof a miss at thr 0.5
    col = np.r_[np.ones(16), np.zeros(4), np.ones(5)]
    assert v3.new_catches(y, cur, 0.5, col, 0.5) == 16
    col[15] = 0.0
    assert v3.new_catches(y, cur, 0.5, col, 0.5) == 15  # below the bar of 16


def test_corrective_share_half_fails_and_ties():
    y = np.array([0, 0, 1, 1])  # two FAs, two misses at thr 0.5
    cur = np.array([0.9, 0.9, 0.1, 0.1])
    r_m1b = np.array([0.5, 0.5, 0.5, 0.5])
    share = v3.corrective_share(y, cur, 0.5, np.array([0.4, 0.6, 0.6, 0.4]), r_m1b)
    assert share == 0.5 and not share > v3.CORRECTIVE_MIN
    assert v3.corrective_share(y, cur, 0.5, np.array([0.5, 0.5, 0.5, 0.5]), r_m1b) == 0.0  # ties never count
    assert v3.corrective_share(y, cur, 0.5, np.array([0.4, 0.4, 0.6, 0.4]), r_m1b) == 0.75


def test_corrective_share_no_errors_is_nan():
    assert np.isnan(v3.corrective_share(np.array([0, 1]), np.array([0.1, 0.9]), 0.5, np.zeros(2), np.zeros(2)))


# ------------------------------------------------------------------ bootstrap and bake-off
def _boot_fixture():
    rng = np.random.default_rng(5)
    spk = np.repeat([f"s{i}" for i in range(12)], 20)
    y = np.tile(np.r_[np.zeros(10), np.ones(10)], 12)
    s = y + rng.normal(0, 0.8, y.size)
    return y, s, s + rng.normal(0, 0.3, y.size), spk


def test_bootstrap_deterministic():
    y, a, b, spk = _boot_fixture()
    r1 = v3.cluster_bootstrap(y, a, b, spk, n=50)
    r2 = v3.cluster_bootstrap(y, a, b, spk, n=50)
    assert r1 == r2 and r1["n_groups"] == 12


def test_bootstrap_resamples_whole_speakers_paired():
    y, a, b, spk = _boot_fixture()
    r = v3.cluster_bootstrap(y, a, b, spk, n=20, return_draws=True)
    assert all(len(d) == 12 for d in r["draws"])  # one draw per speaker slot, not per row
    # identical rules -> every paired Δ is exactly 0 (the same rows feed both)
    same = v3.cluster_bootstrap(y, a, a, spk, n=20)
    assert same["brief_p5"] == 0.0 and same["averse_p5"] == 0.0


def test_bootstrap_missing_speaker_raises():
    y, a, b, spk = _boot_fixture()
    spk = spk.astype(object)
    spk[3] = None
    with pytest.raises(ValueError):
        v3.cluster_bootstrap(y, a, b, spk, n=5)


def _cells10(v=0.0):
    return {f"{k}_{c}": v for k in ("none", "mp3", "noise20", "speed", "shift1") for c in ("brief", "averse")}


@pytest.mark.parametrize("standing,pert,boot,verdict", [
    (True, _cells10(0.0), {"brief_p5": 0.001, "averse_p5": 0.001}, "PASS"),
    (False, _cells10(0.0), {"brief_p5": 0.001, "averse_p5": 0.001}, "FAIL"),
    (True, {**_cells10(0.0), "mp3_brief": -0.0101}, {"brief_p5": 0.001, "averse_p5": 0.001}, "FAIL"),
    (True, {**_cells10(0.0), "mp3_brief": -0.010}, {"brief_p5": 0.001, "averse_p5": 0.001}, "PASS"),
    (True, _cells10(0.0), {"brief_p5": 0.001, "averse_p5": 0.0}, "FAIL"),
])
def test_bakeoff_parts(standing, pert, boot, verdict):
    assert v3.bakeoff_verdict({"ok": standing}, pert, 0.0, boot)["verdict"] == verdict


def test_bakeoff_tripwire_invalid():
    ok_boot = {"brief_p5": 1.0, "averse_p5": 1.0}
    assert v3.bakeoff_verdict({"ok": True}, _cells10(0.0), 1e-6, ok_boot)["verdict"] == "INVALID"
    assert v3.bakeoff_verdict({"ok": True}, None, None, ok_boot)["verdict"] == "INVALID"


def test_perturb_eval_tripwire_and_rounding():
    refs = {"m1b_v3": [0.0, 1.0, 2.0, 3.0], "handcrafted_v5": [0.0, 1.0, 2.0, 3.0], "m5_xlsr_ft": [0.0, 1.0, 2.0, 3.0]}
    rng = np.random.default_rng(7)
    n = 40
    df = pd.DataFrame({"kind": np.repeat(["none", "mp3"], n // 2), "label": np.tile(["bonafide", "spoof"], n // 2),
                       **{k: rng.uniform(0, 3, n) for k in refs}, "spectra_aasist": rng.normal(-3, 2, n)})
    ranks = {k: v3.rank_vs(np.array(refs[k]), df[k].to_numpy()) for k in refs}
    df["fused"] = v3.apply_tiers(v3.blend(ranks, v3.CURRENT_W), df.spectra_aasist.to_numpy(), v3.E_TIERS)
    pe = v3.perturb_eval(df, {"rank_ref_inner_oof_sorted": refs}, {"CURRENT": (v3.CURRENT_W, v3.E_TIERS)})
    assert pe["tripwire"] == 0.0
    assert all(round(v, 4) == v for v in pe["cells"]["CURRENT"].values())
    df["fused"] += 1e-6
    assert v3.perturb_eval(df, {"rank_ref_inner_oof_sorted": refs},
                           {"CURRENT": (v3.CURRENT_W, v3.E_TIERS)})["tripwire"] > v3.TRIPWIRE


# ------------------------------------------------------------------ test agreement and validation
def _shipped(n=6):
    return pd.Series(np.linspace(0.1, 0.9, n), index=[f"HGT{i}.wav" for i in range(n)])


def test_test_agreement_counts_and_pinned():
    sh = _shipped()
    sh.iloc[0] = 0.0005  # a pinned row is excluded
    paths = [f"/x/{n}" for n in sh.index]
    fused = np.linspace(0.0, 1.0, 6)
    ta = v3.test_agreement(paths, fused, (1.0, 0.0), sh)
    assert ta["n_pinned"] == 1 and ta["n_nonpinned"] == 5 and ta["spearman"] == pytest.approx(1.0)
    assert set(ta) == {"spearman", "crossings", "n_nonpinned", "n_pinned"}  # counts only, never names


def test_test_agreement_incomplete_or_duplicate_raises():
    sh = _shipped()
    with pytest.raises(ValueError):
        v3.test_agreement([f"/x/{n}" for n in sh.index[:-1]], np.zeros(5), (1.0, 0.0), sh)
    with pytest.raises(ValueError):
        v3.test_agreement([f"/x/{sh.index[0]}"] * 6, np.zeros(6), (1.0, 0.0), sh)


def _raw_fixture():
    folds = pd.Series({"/a/in0.wav": "0", "/a/in1.wav": "3", "/a/ho.wav": "holdout"})
    raw = pd.DataFrame({"path": ["/a/in0.wav", "/a/in1.wav", "/a/ho.wav", "/t/HGT0.wav", "/i/w0.wav"],
                        "split": ["inner_oof", "inner_oof", "holdout", "test", "itw"],
                        "logit": [0.1, 0.2, 0.3, 0.4, 0.5]})  # fmt: skip
    return raw, folds, {"HGT0.wav"}, {"/i/w0.wav"}


def test_validate_export_clean():
    raw, folds, tn, iw = _raw_fixture()
    assert v3.validate_export(raw, folds, tn, iw) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: pd.concat([r, r.iloc[[0]]]), "duplicate"),
    (lambda r: r.assign(logit=[0.1, np.inf, 0.3, 0.4, 0.5]), "non-finite"),
    (lambda r: r.assign(split=["inner_oof", "holdout", "holdout", "test", "itw"]), "disagrees"),
    (lambda r: r[r.split != "test"], "test coverage"),
    (lambda r: r[r.split != "itw"], "itw coverage"),
])
def test_validate_export_failures(mutate, needle):
    raw, folds, tn, iw = _raw_fixture()
    probs = v3.validate_export(mutate(raw), folds, tn, iw)
    assert any(needle in p for p in probs)


# ------------------------------------------------------------------ outputs and hygiene
def test_write_constants_refuses_existing(tmp_path):
    p = tmp_path / "fusion_v3" / "constants.json"
    v3.write_constants(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    with pytest.raises(FileExistsError):
        v3.write_constants(p, {"a": 2})


def test_write_target_is_fusion_v3():
    assert v3.FUSION_V3 == REPO / "models" / "fusion_v3" / "constants.json"
    assert "fusion_v2" not in str(v3.FUSION_V3)


def test_write_refused_without_ratification():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "fuse_sweep_v3.py"), "--write"], capture_output=True,
                       text=True, cwd=REPO, check=False)  # fmt: skip
    assert r.returncode != 0 and "--ratified-by nathan" in r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "fuse_sweep_v3.py"), "--write", "--ratified-by", "me",
                        "--candidate", "W4"], capture_output=True, text=True, cwd=REPO, check=False)  # fmt: skip
    assert r.returncode != 0


def test_sha_mismatch_detected(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("filename\tcm-score\n")
    assert not v3.sha_ok(p)


def test_self_check_failure_invalidates_everything():
    rep = v3.finalize({"candidates": {}}, self_check_ok=False)
    assert all(c["status"] == "INVALID" for c in rep["candidates"].values())
    assert set(rep["candidates"]) == set(v3.CANDIDATES) and rep["decision"]["pick"] is None


def test_finalize_ratifiable_only():
    rep = {"candidates": {"T2": {"ratifiable": False}, "W4": {"ratifiable": False, "status": "BAKEOFF FAIL"},
                          "P_wl": {"status": "NOT RUN"}, "H_noise": {"status": "INVALID"}}}  # fmt: skip
    out = v3.finalize(rep, self_check_ok=True)
    assert out["decision"]["pick"] is None and out["decision"]["line"].startswith("KEEP")
    rep["candidates"]["W4"] = {"ratifiable": True, "readout": {c: 0.1 for c in v3.CELLS}}
    assert v3.finalize(rep, self_check_ok=True)["decision"]["pick"] == "W4"


def test_import_hygiene():
    code = ("import importlib.util,sys;s=importlib.util.spec_from_file_location('m','scripts/fuse_sweep_v3.py');"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "print('lightgbm' in sys.modules, 'torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=False)
    assert r.stdout.strip() == "False False", r.stderr


# ------------------------------------------------------------------ needs_data self-checks
@pytest.fixture(scope="module")
def rep():
    if not HAVE_DATA:
        pytest.skip("needs detector exports and the shipped TSV")
    return v3.evaluate()


@pytest.mark.needs_data
def test_self_check_reproduces_shipped_numbers(rep):
    assert rep["self_check"]["ok"], rep["self_check"]
    assert rep["rows_base"] == {"inner_oof": 16142, "holdout": 3858, "test": 1671, "itw": 3000}
    assert rep["current"]["test"]["n_nonpinned"] == 1671  # the shipped file's pinned block is empty


@pytest.mark.needs_data
def test_t2_identical_cells_and_w4_known_numbers(rep):
    t2, w4 = rep["candidates"]["T2"], rep["candidates"]["W4"]
    assert t2["known_number_check"]["six_identical_to_current"]
    assert t2["diagnostic"]["ok"]  # zero spoof fires everywhere
    assert all(w4["known_number_check"].values())


@pytest.mark.needs_data
def test_perturbation_tripwire(rep):
    assert rep["candidates"]["W4"]["bakeoff"]["tripwire_max_abs"] <= v3.TRIPWIRE


@pytest.mark.needs_data
def test_missing_export_not_run_others_evaluated(rep):
    if (v3.S / "handcrafted_v6.csv").exists():
        pytest.skip("handcrafted_v6 export present")
    assert rep["candidates"]["H_noise"]["status"] == "NOT RUN"
    assert "readout" in rep["candidates"]["T2"] and "readout" in rep["candidates"]["W4"]


# ------------------------------------------------------------------ hermetic synthetic world (audit round 1)
def _world(tmp_path, monkeypatch, seed=0):
    """A small labelled world with every split, the base columns, M3, a shipped file and a perturbation
    cohort; v3.S points at an empty tmp dir so new-column exports are whatever a test writes there."""
    rng = np.random.default_rng(seed)
    recs = []
    for split, n in (("inner_oof", 200), ("holdout", 100), ("test", 30), ("itw", 80)):
        for i in range(n):
            y = i % 2
            p = f"/w/HGT{i}.wav" if split == "test" else f"/w/{split}{i}.wav"
            src = ("ljspeech" if i % 4 == 0 else "librispeech") if y == 0 else "diffssd"
            fold = "holdout" if split == "holdout" else (str(i % 5) if split == "inner_oof" else "")
            recs.append({"path": p, "split": split, "y": y, "source": src, "speaker": f"s{i % 8}", "fold": fold})
    w = pd.DataFrame(recs)
    labels = w.assign(label=np.where(w.y == 1, "spoof", "bonafide")).set_index("path")[["label", "source", "speaker"]]
    folds = w[w.split.isin(["inner_oof", "holdout"])].set_index("path").fold

    def col(scale, noise):
        return w.assign(logit=scale * w.y + rng.normal(0, noise, len(w))).set_index("path")[["split", "logit"]]

    base_cols = {"m1b_v3": col(2.0, 1.0), "handcrafted_v5": col(1.0, 1.0), "m5_xlsr_ft": col(1.5, 1.0)}
    m3 = w.assign(logit=np.where(w.y == 1, 2.0, -5.0) + rng.normal(0, 2.0, len(w))).set_index("path")[["split", "logit"]]
    base = v3.build(base_cols, m3, labels)
    shipped = pd.Series(rng.uniform(0.01, 1.0, 30), index=[f"HGT{i}.wav" for i in range(30)])
    consts = {"rank_ref_inner_oof_sorted": {k: base["ref"][k].tolist() for k in base_cols}}
    pk = pd.DataFrame({"kind": np.repeat(["none", "mp3"], 40), "label": np.tile(["bonafide", "spoof"], 40)})
    for k in base_cols:
        pk[k] = rng.normal(0, 1.5, len(pk))
    pk["spectra_aasist"] = rng.normal(-2, 1.5, len(pk))
    ranks = {k: v3.rank_vs(base["ref"][k], pk[k].to_numpy()) for k in base_cols}
    pk["fused_base"] = v3.blend(ranks, v3.CURRENT_W)
    pk["fused"] = v3.apply_tiers(pk.fused_base.to_numpy(), pk.spectra_aasist.to_numpy(), v3.E_TIERS)
    monkeypatch.setattr(v3, "S", tmp_path)
    world = {"base": base, "base_cols": base_cols, "m3": m3, "labels": labels, "folds": folds, "shipped": shipped,
             "perturb": pk, "consts": consts, "test_names": set(shipped.index), "itw_paths": set(base["idx"]["itw"]),
             "inputs": {}}  # fmt: skip
    return world, w


def _export(w, tmp_path, name, sign=1.0, drop_inner=0, itw_separate=False, seed=9):
    rng = np.random.default_rng(seed)
    d = w.assign(logit=sign * (1.8 * w.y + rng.normal(0, 1.0, len(w))), score=0.5)[["path", "fold", "split", "score",
                                                                                     "logit"]]  # fmt: skip
    if drop_inner:
        d = d.drop(d.index[d.split == "inner_oof"][:drop_inner])
    if itw_separate:
        d[d.split == "itw"][["path", "logit"]].to_csv(tmp_path / f"_itw_{name}.csv", index=False)
        d = d[d.split != "itw"]
    d.to_csv(tmp_path / f"{name}.csv", index=False)


def test_world_not_run_and_others_evaluated(tmp_path, monkeypatch):
    world, _ = _world(tmp_path, monkeypatch)
    out = v3.run_candidates(world)
    assert out["P_wl"]["status"] == "NOT RUN" and out["H_noise"]["status"] == "NOT RUN"
    assert "readout" in out["T2"] and "bakeoff" in out["W4"]
    assert out["W4"]["gate"]["governing"] is False and out["T2"]["gate"]["governing"] is True


def test_world_direction_guard_and_catch_wiring(tmp_path, monkeypatch):
    world, w = _world(tmp_path, monkeypatch)
    _export(w, tmp_path, "wavlm_l", sign=-1.0)
    bad = v3.run_candidates(world)["P_wl"]
    assert bad["status"] == "INVALID" and "AUC" in bad["reason"]
    _export(w, tmp_path, "wavlm_l", sign=1.0)
    good = v3.run_candidates(world)["P_wl"]
    assert "readout" in good and good["status"] in {"PASS WITH ROOM", "GATE PASS, NO ROOM", "FAIL"}
    # catch wiring: misses at CURRENT's inner-OOF threshold, catches at the column's own inner-OOF threshold
    ctx = world["base"]
    _, cur = v3.score(ctx, v3.CURRENT_W, v3.E_TIERS)
    thr = v3.inner_threshold(ctx["y"]["inner_oof"], cur["inner_oof"])
    assert good["diagnostic"]["itw_misses_at_current_inner_thr"] == int(
        np.sum((ctx["y"]["itw"] == 1) & (cur["itw"] < thr)))
    col = pd.read_csv(tmp_path / "wavlm_l.csv").set_index("path")
    ci, cw = col.loc[ctx["idx"]["inner_oof"], "logit"].to_numpy(), col.loc[ctx["idx"]["itw"], "logit"].to_numpy()
    assert good["diagnostic"]["new_catches"] == v3.new_catches(ctx["y"]["itw"], cur["itw"], thr, cw,
                                                                v3.inner_threshold(ctx["y"]["inner_oof"], ci))


def test_world_broken_export_is_invalid_not_fatal(tmp_path, monkeypatch):
    world, w = _world(tmp_path, monkeypatch)
    w.assign(logit=0.0)[["path", "logit"]].to_csv(tmp_path / "wavlm_l.csv", index=False)  # no split column
    out = v3.run_candidates(world)
    assert out["P_wl"]["status"] == "INVALID" and "lacks columns" in out["P_wl"]["reason"]
    assert "readout" in out["T2"] and "readout" in out["W4"]
    (tmp_path / "wavlm_l.csv").write_bytes(b"\x00\x01garbage")
    assert v3.run_candidates(world)["P_wl"]["status"] == "INVALID"


def test_world_h_noise_fails_not_invalid(tmp_path, monkeypatch):
    world, w = _world(tmp_path, monkeypatch)
    _export(w, tmp_path, "handcrafted_v6")  # in-file ITW rows but the declared _itw file is absent
    assert v3.run_candidates(world)["H_noise"]["status"] == "INVALID"
    _export(w, tmp_path, "handcrafted_v6", itw_separate=True)
    h = v3.run_candidates(world)["H_noise"]
    assert h["status"] == "FAIL" and h["diagnostic"]["ok"] is False and "5_diagnostic" in h["gate"]["failed"]


def test_world_shrunk_rows_rebuild_refs(tmp_path, monkeypatch):
    world, w = _world(tmp_path, monkeypatch)
    _export(w, tmp_path, "wavlm_l", drop_inner=10)
    p = v3.run_candidates(world)["P_wl"]
    assert p["rows_shrunk_vs_base"] and p["rows"]["inner_oof"] == 190 and world["base"]["rows"]["inner_oof"] == 200
    ctx = v3.build({**world["base_cols"], "wavlm_l": v3.load_raw("wavlm_l", None).set_index("path")}, world["m3"],
                   world["labels"])  # fmt: skip
    assert len(ctx["ref"]["m1b_v3"]) == 190  # rank references from the shrunk context's own inner rows


def test_world_separate_itw_file_picked_up(tmp_path, monkeypatch):
    world, w = _world(tmp_path, monkeypatch)
    _export(w, tmp_path, "wavlm_l", itw_separate=True)
    p = v3.run_candidates(world)["P_wl"]
    assert "readout" in p and p["rows"]["itw"] == 80


def test_world_rule7_common_rows_and_fallback(tmp_path, monkeypatch):
    world, w = _world(tmp_path, monkeypatch)
    _export(w, tmp_path, "wavlm_l", drop_inner=10)
    cells = {c: 0.1 for c in v3.CELLS}
    rep = {"candidates": {"W4": {"ratifiable": True, "readout": cells}, "P_wl": {"ratifiable": True, "readout": cells}}}
    out = v3.finalize(rep, True, ctxs=world)
    assert out["rule7_common_rows"]["rows"]["inner_oof"] == 190  # both passers compared on the same rows
    assert set(out["rule7_common_rows"]["cells"]) == {"W4", "P_wl"}
    (tmp_path / "wavlm_l.csv").unlink()
    rep2 = {"candidates": {"W4": {"ratifiable": True, "readout": cells}, "P_wl": {"ratifiable": True, "readout": cells}}}
    out2 = v3.finalize(rep2, True, ctxs=world)
    assert out2["decision"]["pick"] is None and "rule7_error" in out2 and out2["decision"]["line"].startswith("KEEP")


def test_world_t2_diagnostic_counterfactual(tmp_path, monkeypatch):
    world, _ = _world(tmp_path, monkeypatch)
    pk = world["perturb"].copy()
    pk.loc[pk.index[1], ["spectra_aasist", "fused_base"]] = [-7.0, 0.8]  # one spoof row in kind 'none'
    world["perturb"] = pk
    t2 = v3.run_candidates(world)["T2"]
    assert t2["diagnostic"]["deeper_tier_fires"]["perturb_none"]["spoof"] == 1
    assert t2["diagnostic"]["ok"] is False and "5_diagnostic" in t2["gate"]["failed"]


def test_pwl_diag_ok_composition():
    assert v3.pwl_diag_ok(16, {"inner_oof": 0.6, "itw": 0.6})
    assert not v3.pwl_diag_ok(15, {"inner_oof": 0.6, "itw": 0.6})
    assert not v3.pwl_diag_ok(16, {"inner_oof": 0.5, "itw": 0.6})
    assert not v3.pwl_diag_ok(16, {"inner_oof": 0.6, "itw": float("nan")})


def test_room_itw_brief_side():
    d = _cells(itw_brief=0.0299, itw_averse=0.03)
    assert v3.room(d, 19)["failed"] == ["itw_brief_gain_0.030"]


def test_test_agreement_same_length_unmatched_raises():
    sh = _shipped()
    paths = [f"/x/{n}" for n in sh.index[:-1]] + ["/x/NOTSHIPPED.wav"]
    with pytest.raises(ValueError):
        v3.test_agreement(paths, np.zeros(6), (1.0, 0.0), sh)


def test_clean_makes_json_safe():
    out = v3.clean({"a": float("nan"), "b": [np.float64(1.5), np.bool_(True), np.int64(3)], "c": (np.inf,)})
    assert out == {"a": None, "b": [1.5, True, 3], "c": [None]}
    json.dumps(out, allow_nan=False)


class _Args:
    def __init__(self, candidate="W4", sha="x", model=None):
        self.candidate, self.report_sha256, self.new_column_model, self.ratified_by = candidate, sha, model, "nathan"


def test_precheck_write_refusals(tmp_path, monkeypatch):
    rp = tmp_path / "rep.json"
    monkeypatch.setattr(v3, "REPORT", rp)
    with pytest.raises(SystemExit, match="no locked report"):
        v3.precheck_write(_Args())
    rp.write_text(json.dumps({"decision": {"pick": "P_wl"}, "inputs": {}}))
    sha = v3.sha256(rp)
    with pytest.raises(SystemExit, match="not the locked one"):
        v3.precheck_write(_Args(sha="0" * 64))
    with pytest.raises(SystemExit, match="not the locked pick"):
        v3.precheck_write(_Args(candidate="W4", sha=sha))
    with pytest.raises(SystemExit, match="new-column-model"):
        v3.precheck_write(_Args(candidate="P_wl", sha=sha))
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / "meta.json").write_text("{}")
    assert v3.precheck_write(_Args(candidate="P_wl", sha=sha, model=str(tmp_path / "probe")))["decision"]["pick"] == "P_wl"
    rp.write_text(json.dumps({"decision": {"pick": "W4"}, "inputs": {}}))
    with pytest.raises(SystemExit, match="not used by W4"):
        v3.precheck_write(_Args(candidate="W4", sha=v3.sha256(rp), model=str(tmp_path / "probe")))


def test_do_write_refuses_changed_inputs_or_decision():
    locked = {"inputs": {"a": "1"}, "decision": {"pick": "W4", "passers": ["W4"], "line": "x"}}
    with pytest.raises(SystemExit, match="inputs changed"):
        v3.do_write(_Args(), {"inputs": {"a": "2"}, "decision": locked["decision"]}, locked)
    with pytest.raises(SystemExit, match="decision differs"):
        v3.do_write(_Args(), {"inputs": {"a": "1"}, "decision": {**locked["decision"], "pick": None}}, locked)


def test_main_exits_2_on_self_check_failure(tmp_path, monkeypatch):
    rep = v3.finalize({"candidates": {}, "rows_base": {}, "self_check": {"ok": False, "got": {}, "spearman": 0.0},
                       "current": {}}, self_check_ok=False)  # fmt: skip
    monkeypatch.setattr(v3, "evaluate", lambda: rep)
    monkeypatch.setattr(v3, "sha_ok", lambda *a, **k: True)
    monkeypatch.setattr(v3, "REPORT", tmp_path / "sweep_v3_report.json")
    with pytest.raises(SystemExit) as e:
        v3.main([])
    assert e.value.code == 2 and (tmp_path / "sweep_v3_report.json").exists()
