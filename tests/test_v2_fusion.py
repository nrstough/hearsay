"""Tests for the V2-60 fusion test (src/hearsay/fusion_v2.py, scripts/v2_fusion.py).

Run spec: docs/specs/2026-10-11_v2-benchmark-fusion.md (sections L (fusion part), O, F);
pre-declaration: benchmarks/v2_60_candidates.json.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hearsay import benchmark as B
from hearsay import fusion_v2 as F

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v2_fusion", REPO / "scripts/v2_fusion.py")
VF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VF)

CAND = REPO / "benchmarks/v2_60_candidates.json"
SPEC = REPO / "docs/specs/2026-10-11_v2-benchmark-fusion.md"
HAVE_DATA = (REPO / "outputs/detector_scores/m1b_v3.csv").exists() and (REPO / "outputs/v2/benchmark/scores/xlsr_L20.csv").exists()
def needs_data(f):
    return pytest.mark.needs_data(pytest.mark.skipif(not HAVE_DATA, reason="needs the gitignored score exports")(f))
TARGET = np.array([0.6, 0.2, 0.2, 0.0, 0.0])
COLS = ["m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "wavlm_l", "xlsr_L20"]


@pytest.fixture(autouse=True)
def _tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARSAY_HOLDOUT_LEDGER", str(tmp_path / "ledger"))


def world(seed=0, n_per_fold=120, noise_col=False):
    """Five folds, five columns: three informative, two weak; optional pure noise in column 4."""
    rng = np.random.default_rng(seed)
    n = 5 * n_per_fold
    fold = np.array([str(i % 5) for i in range(n)])
    y = (rng.random(n) < 0.5).astype(int)
    strength = [1.6, 0.9, 1.2, 0.4, 0.0 if noise_col else 0.3]
    L = np.column_stack([s * y + rng.normal(0, 1, n) for s in strength])
    spectra = np.where(y == 1, rng.normal(3, 1, n), rng.normal(-1, 2, n))
    return L, y, fold, spectra


# ---------------------------------------------------------------------------
# L. Leakage (fusion part)
# ---------------------------------------------------------------------------


def test_alpha_selection_never_sees_outer_fold():
    """Poison the outer fold's logits and labels: the alpha chosen for it is unchanged."""
    L, y, fold, _ = world(1)
    tr = fold != "2"
    a1, c1 = F.select_alpha(L, y, fold, tr, TARGET)
    L2, y2 = L.copy(), y.copy()
    L2[fold == "2"] = 100.0 * np.random.default_rng(9).normal(size=L2[fold == "2"].shape)
    y2[fold == "2"] = 1 - y2[fold == "2"]
    a2, c2 = F.select_alpha(L2, y2, fold, tr, TARGET)
    assert a1 == a2 and c1 == c2


def test_inner_inner_refs_exclude_outer_and_inner_fold():
    """Within inner fold g, a row's ranks depend only on itself and the other
    training folds: poisoning the rest of g (and the outer fold) cannot move them."""
    L, y, fold, _ = world(2)
    tr = fold != "0"
    blocks1 = {g: X for g, X, _, _ in F.inner_inner_blocks(L, y, fold, tr)}
    L2 = L.copy()
    g_rows = np.flatnonzero(fold == "3")
    L2[g_rows[1:]] = -50.0
    L2[fold == "0"] = 50.0
    blocks2 = {g: X for g, X, _, _ in F.inner_inner_blocks(L2, y, fold, tr)}
    assert np.array_equal(blocks1["3"][0], blocks2["3"][0])


def test_c2_fit_inputs_are_train_folds_only():
    """Flip the outer fold's labels and scramble its logits: that fold's fitted weights do not move."""
    L, y, fold, spectra = world(3)
    r1 = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    L2, y2 = L.copy(), y.copy()
    m = fold == "1"
    y2[m] = 1 - y2[m]
    L2[m] = np.random.default_rng(4).normal(size=L2[m].shape) * 9
    r2 = F.run_c2_nested(L2, y2, fold, spectra, COLS, TARGET)
    assert np.array_equal(r1["folds"]["1"]["weights"], r2["folds"]["1"]["weights"])
    assert r1["folds"]["1"]["alpha"] == r2["folds"]["1"]["alpha"]
    assert not np.array_equal(r1["folds"]["0"]["weights"], r2["folds"]["0"]["weights"])


def test_spectra_refused_as_fit_feature():
    L, y, fold, spectra = world(5)
    with pytest.raises(B.BenchmarkError, match="never be a fit feature"):
        F.run_c2_nested(L, y, fold, spectra, ["m1b_v3", "spectra_aasist", "a", "b", "c"], TARGET)
    with pytest.raises(B.BenchmarkError):
        F.fit_c2_offinner(L, y, fold, ["spectra_aasist", "a", "b", "c", "d"], TARGET)


def test_alpha_invariant_to_spectra_column():
    """K3: Spectra never touches alpha or the weights; permuting it changes only E."""
    L, y, fold, spectra = world(6)
    r1 = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    r2 = F.run_c2_nested(L, y, fold, np.random.default_rng(1).permutation(spectra) - 2.0, COLS, TARGET)
    for f in r1["folds"]:
        assert r1["folds"][f]["alpha"] == r2["folds"][f]["alpha"]
        assert np.array_equal(r1["folds"][f]["weights"], r2["folds"][f]["weights"])
    assert np.array_equal(r1["base"], r2["base"])


def test_run_refuses_if_candidates_differ_from_spec_hash(tmp_path):
    F.check_candidates(CAND, SPEC)
    spec = SPEC.read_text()
    bad_spec = tmp_path / "spec.md"
    bad_spec.write_text(spec.replace("brief 0.02, package-0.5 0.05", "brief 0.03, package-0.5 0.05", 1))
    with pytest.raises(B.BenchmarkError, match="frozen spec region changed"):
        F.check_candidates(CAND, bad_spec)
    cand = json.loads(CAND.read_text())
    cand["candidates"]["C2"]["alpha_grid"] = [0.0, 0.5, 1.0]
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps(cand))
    with pytest.raises(B.BenchmarkError, match="C2 settings"):
        F.check_candidates(bad, SPEC)


def test_spec_hash_ignores_results_section_edits(tmp_path):
    spec = SPEC.read_text()
    edited = tmp_path / "spec.md"
    edited.write_text(spec.replace("## Results\n", "## Results\n\nC2 TIE, recorded after the run.\n") + "\nmore\n")
    F.check_candidates(CAND, edited)


def test_candidates_json_matches_frozen_spec_region():
    cand = json.loads(CAND.read_text())
    region = F.frozen_region(SPEC.read_text(), cand["spec_frozen_region"]["start"], cand["spec_frozen_region"]["end"])
    assert "Tie bands:** brief 0.02, package-0.5 0.05, package-0.3 0.03" in region
    assert cand["tie_bands"] == {"brief": 0.02, "package05": 0.05, "package03": 0.03}
    assert "f4 package-0.5 ≤ C0's f4 package-0.5 + 0.05" in region and cand["f4_guard"]["max_above_control"] == 0.05
    assert "0.6·m1b_v3 + 0.2·handcrafted_v5 + 0.2·m5_xlsr_ft" in region
    assert cand["candidates"]["C0"]["weights"] == {"m1b_v3": 0.6, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2,
                                                   "wavlm_l": 0.0, "xlsr_L20": 0.0}
    assert "Equal rank weights (0.2 each)" in region and set(cand["candidates"]["C1"]["weights"].values()) == {0.2}
    m = re.search(r"\| C3 \| ([^|]+)\|", region)
    assert m and "0.48·m1b_v3 + 0.16·hc_v5 + 0.16·m5 + 0.1·wavlm_l + 0.1·xlsr_L20" in m.group(1)
    assert cand["candidates"]["C3"]["weights"] == {"m1b_v3": 0.48, "handcrafted_v5": 0.16, "m5_xlsr_ft": 0.16,
                                                   "wavlm_l": 0.1, "xlsr_L20": 0.1}
    assert "α ∈ {0, .25, .5, .75}" in region and "π_eff 0.8" in region
    for c in cand["candidates"].values():
        if c["kind"] == "fixed":
            assert abs(sum(c["weights"].values()) - 1.0) < 1e-12


def test_candidates_json_must_be_committed(tmp_path):
    F.committed_blob_sha(CAND)
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"],
                   cwd=repo, check=True)
    f = repo / "c.json"
    f.write_text("{}")
    with pytest.raises(B.BenchmarkError, match="not committed"):
        F.committed_blob_sha(f, repo)
    subprocess.run(["git", "add", "c.json"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "c"], cwd=repo, check=True)
    F.committed_blob_sha(f, repo)
    f.write_text('{"a": 1}')
    with pytest.raises(B.BenchmarkError, match="uncommitted"):
        F.committed_blob_sha(f, repo)


def test_v2_fusion_refuses_holdout_for_rule_not_in_json():
    cand = json.loads(CAND.read_text())
    assert [d["name"] for d in VF.holdout_defs(cand, ["C0", "C2"])] == ["C0", "C2"]
    with pytest.raises(B.BenchmarkError, match="not in the committed candidates"):
        VF.holdout_defs(cand, ["C0", "C4"])


def test_verdict_refuses_stale_scorecard_hashes():
    ro = {"candidates_blob_sha": "a", "registry_sha256": "r"}
    F.check_readout_fresh(ro, candidates_blob="a", registry_sha="r")
    with pytest.raises(B.BenchmarkError, match="candidates"):
        F.check_readout_fresh(ro, candidates_blob="b", registry_sha="r")
    with pytest.raises(B.BenchmarkError, match="registry"):
        F.check_readout_fresh(ro, candidates_blob="a", registry_sha="s")


# ---------------------------------------------------------------------------
# O. Overfitting
# ---------------------------------------------------------------------------


def test_c2_reports_in_sample_gap():
    L, y, fold, spectra = world(7)
    r = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    for d in r["folds"].values():
        for k in B.READINGS:
            assert d["gap"][k] == pytest.approx(d["oof_fold"][k] - d["in_sample"][k])


def _vin(pooled, f4, basis=F.BASIS):
    return {"basis": basis, "pooled": pooled, "f4_package05": f4}


def test_verdict_uses_nested_oof_only():
    p = {"brief": 0.15, "package05": 0.30, "package03": 0.2}
    with pytest.raises(B.BenchmarkError, match="nested inner OOF only"):
        F.verdict(_vin(p, 0.1), _vin(p, 0.1, basis="in_sample"), {"brief": .02, "package05": .05, "package03": .03}, 0.05)
    with pytest.raises(B.BenchmarkError, match="nested inner OOF only"):
        F.verdict(_vin(p, 0.1, basis="transductive"), _vin(p, 0.1), {"brief": .02, "package05": .05, "package03": .03}, 0.05)


def test_noise_column_gets_near_zero_weight_after_shrink():
    L, y, fold, _ = world(8, n_per_fold=300, noise_col=True)
    fit = F.fit_c2(L, y, fold, np.ones(len(y), dtype=bool), TARGET)
    assert fit["w_fit_normalized"][4] < 0.05
    assert fit["weights"][4] < 0.05


def test_c2_reports_effective_params_and_alpha_per_fold():
    L, y, fold, spectra = world(9)
    r = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    for d in r["folds"].values():
        nnz = int(np.count_nonzero(d["w_fit_raw"] > 0))
        assert d["effective_params"] == pytest.approx(1 + (1 - d["alpha"]) * (nnz - 1))
        assert d["alpha"] in F.ALPHA_GRID and {"w_fit_l1", "intercept", "success", "nit"} <= set(d)
    assert F.effective_params(0.75, np.array([1.0, 0, 2.0, 0, 0])) == 1.25


def test_alpha_per_fold_reported_and_spread_flagged():
    L, y, fold, spectra = world(10)
    r = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    alphas = [d["alpha"] for d in r["folds"].values()]
    assert r["alpha_spread"] == max(alphas) - min(alphas)
    assert r["alpha_at_grid_max_all"] == all(a == max(F.ALPHA_GRID) for a in alphas)


def test_alpha_at_grid_edge_flagged():
    L, y, fold, _ = world(11)
    fit = F.fit_c2(L, y, fold, np.ones(len(y), dtype=bool), TARGET)
    assert fit["alpha_at_grid_max"] == (fit["alpha"] == 0.75)


def test_learned_readout_includes_holdout_and_itw_gap():
    inner = {"brief": 0.15, "package05": 0.33, "package03": 0.21}
    g = VF.population_gap(inner, {"brief": 0.01, "package05": 0.02, "package03": None})
    assert g == {"brief": pytest.approx(-0.14), "package05": pytest.approx(-0.31), "package03": None}
    assert VF.population_gap(inner, {}) == {"brief": None, "package05": None, "package03": None}


def test_followon_refused_unless_c2_verdict_win():
    assert F.followon_allowed("WIN (provisional)") and F.followon_allowed("WIN")
    assert not F.followon_allowed("TIE") and not F.followon_allowed("LOSE")


# ---------------------------------------------------------------------------
# F. Fit pieces and verdict
# ---------------------------------------------------------------------------


def test_nnlr_weights_nonnegative():
    rng = np.random.default_rng(12)
    y = (rng.random(800) < 0.5).astype(int)
    X = np.column_stack([y + rng.normal(0, 1, 800), -2 * y + rng.normal(0, 1, 800)])  # one anti-informative column
    w, _, info = F.fit_nnlr(X, y, F.sample_weights(y))
    assert (w >= 0).all() and w[1] == 0.0 and w[0] > 0 and info["success"]


def test_c2_prior_is_0_8_for_package05():
    y = np.array([0, 0, 0, 1])
    sw = F.sample_weights(y, 0.8)
    assert sw[y == 0].sum() == pytest.approx(0.2) and sw[y == 1].sum() == pytest.approx(0.8)
    assert F.PRIOR_FAKE == 0.8 == json.loads(CAND.read_text())["candidates"]["C2"]["prior_fake"]
    # pi_eff for package-0.5: 0.5*4 / (0.5*4 + 0.5*1)
    assert 0.5 * 4 / (0.5 * 4 + 0.5 * 1) == F.PRIOR_FAKE


def test_alpha_grid_fixed():
    assert F.ALPHA_GRID == (0.0, 0.25, 0.5, 0.75)
    assert tuple(json.loads(CAND.read_text())["candidates"]["C2"]["alpha_grid"]) == F.ALPHA_GRID


def test_full_shrink_recovers_c0_weights():
    assert np.array_equal(F.shrink(np.array([3.0, 1.0, 0.0, 2.0, 5.0]), 1.0, TARGET), TARGET)


def test_shrink_applied_after_normalization():
    w = F.shrink(np.array([30.0, 10.0]), 0.5, np.array([0.6, 0.4]))
    assert np.allclose(w, 0.5 * np.array([0.6, 0.4]) + 0.5 * np.array([0.75, 0.25]))
    assert w.sum() == pytest.approx(1.0)


def test_zero_sum_fit_weights_raises():
    with pytest.raises(B.BenchmarkError, match="all-zero"):
        F.shrink(np.zeros(5), 0.5, TARGET)


def test_alpha_tie_breaks_to_larger_alpha():
    assert F.choose_alpha({0.0: 0.3, 0.25: 0.3, 0.5: 0.3, 0.75: 0.3}) == 0.75
    assert F.choose_alpha({0.0: 0.2, 0.25: 0.3, 0.5: 0.3, 0.75: 0.3}) == 0.0
    assert F.choose_alpha({0.0: 0.3, 0.25: 0.2, 0.5: 0.2 + 1e-13, 0.75: 0.25}) == 0.5


def _e_world(seed=0, n=120):
    """A world where Spectra fires on half the rows of both classes, so applying
    E inside the alpha search changes the choice (seed 0: pre-E 0.0, post-E 0.75)."""
    rng = np.random.default_rng(seed)
    N = 5 * n
    fold = np.array([str(i % 5) for i in range(N)])
    y = (rng.random(N) < 0.5).astype(int)
    L = np.column_stack([s * y + rng.normal(0, 1, N) for s in [0.8, 1.4, 0.6, 1.0, 0.5]])
    spectra = np.where(rng.random(N) < 0.5, -5.0, 2.0)
    return L, y, fold, spectra


def test_alpha_pre_e_choice_differs_from_post_e_on_fixture():
    L, y, fold, spectra = _e_world(0)
    tr = fold != "0"
    blocks = list(F.inner_inner_blocks(L, y, fold, tr))
    post = {}
    for a in F.ALPHA_GRID:
        s, yy, m3 = [], [], []
        for g, X, yb, w in blocks:
            s.append(F.blend_cols(X, F.shrink(w, a, TARGET)))
            yy.append(yb)
            m3.append(spectra[tr & (fold == g)])
        post[a] = B.reading(np.concatenate(yy), B.apply_e(np.concatenate(s), np.concatenate(m3))[0], "package05")
    a_post = F.choose_alpha(post)
    a_pre, _ = F.select_alpha(L, y, fold, tr, TARGET)
    assert a_pre != a_post
    assert a_pre == 0.0 and a_post == 0.75
    r = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    assert r["folds"]["0"]["alpha"] == a_pre


def test_alpha_selected_on_package05_pre_e():
    """The per-alpha costs are pooled package-0.5 of the pre-E blocks, recomputed independently."""
    L, y, fold, _ = world(13)
    tr = fold != "4"
    alpha, costs = F.select_alpha(L, y, fold, tr, TARGET)
    blocks = list(F.inner_inner_blocks(L, y, fold, tr))
    for a in F.ALPHA_GRID:
        s = np.concatenate([F.blend_cols(X, F.shrink(w, a, TARGET)) for _, X, _, w in blocks])
        yy = np.concatenate([b for _, _, b, _ in blocks])
        assert costs[a] == B.reading(yy, s, "package05")
    assert alpha == F.choose_alpha(costs)
    sig = ast.parse((REPO / "src/hearsay/fusion_v2.py").read_text())
    fn = next(n for n in ast.walk(sig) if isinstance(n, ast.FunctionDef) and n.name == "select_alpha")
    assert "spectra" not in ast.dump(fn) and "apply_e" not in ast.dump(fn)


def test_c2_has_no_platt_step():
    L, y, fold, spectra = world(14)
    r = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    assert r["platt"] is None
    assert np.array_equal(r["fused"], B.apply_e(r["base"], spectra)[0])
    src = (REPO / "src/hearsay/fusion_v2.py").read_text()
    assert "fit_platt" not in src and "LogisticRegression" not in src
    assert json.loads(CAND.read_text())["candidates"]["C2"]["platt"] is False


def test_c2_offinner_model_fit_on_all_inner_only():
    """Poison every non-inner population: the off-inner C2 weights do not move."""
    from types import SimpleNamespace

    L, y, fold, _ = world(15)
    pop = SimpleNamespace(y=y, fold=fold)
    rng = np.random.default_rng(3)

    def bench(scale):
        cols = {c: {"inner": L[:, j], "holdout": scale * rng.normal(size=50), "itw": scale * rng.normal(size=40),
                    "test": scale * rng.normal(size=10)} for j, c in enumerate(COLS)}
        return SimpleNamespace(pops={"inner": pop}, cols=cols)

    w1 = VF.c2_offinner_weights(bench(1.0), COLS, TARGET)["weights"]
    w2 = VF.c2_offinner_weights(bench(1e6), COLS, TARGET)["weights"]
    assert np.array_equal(w1, w2)


def _p(b, p5, p3):
    return {"brief": b, "package05": p5, "package03": p3}


BANDS = {"brief": 0.02, "package05": 0.05, "package03": 0.03}


def test_verdict_win_tie_lose_boundaries():
    c0 = _vin(_p(0.15, 0.3390, 0.21), 0.1649)
    assert F.verdict(c0, _vin(_p(0.15, 0.2889, 0.21), 0.1649), BANDS, 0.05)["verdict"] == "WIN"
    assert F.verdict(c0, _vin(_p(0.15, 0.2890, 0.21), 0.1649), BANDS, 0.05)["verdict"] == "TIE"
    assert F.verdict(c0, _vin(_p(0.15, 0.3890, 0.21), 0.1649), BANDS, 0.05)["verdict"] == "TIE"
    assert F.verdict(c0, _vin(_p(0.15, 0.3891, 0.21), 0.1649), BANDS, 0.05)["verdict"] == "LOSE"
    assert F.verdict(c0, _vin(_p(0.15, 0.2889, 0.21), 0.1649), BANDS, 0.05,
                     label_provisional=True)["verdict"] == "WIN (provisional)"


def test_verdict_any_reading_regression_blocks_win():
    c0 = _vin(_p(0.15, 0.3390, 0.21), 0.1649)
    assert F.verdict(c0, _vin(_p(0.1701, 0.20, 0.21), 0.1649), BANDS, 0.05)["verdict"] == "LOSE"
    assert F.verdict(c0, _vin(_p(0.17, 0.20, 0.24), 0.1649), BANDS, 0.05)["verdict"] == "WIN"
    assert F.verdict(c0, _vin(_p(0.15, 0.20, 0.2401), 0.1649), BANDS, 0.05)["verdict"] == "LOSE"


def test_f4_guard_blocks_win():
    c0 = _vin(_p(0.15, 0.3390, 0.21), 0.1649)
    assert F.verdict(c0, _vin(_p(0.15, 0.20, 0.21), 0.2149), BANDS, 0.05)["verdict"] == "WIN"
    v = F.verdict(c0, _vin(_p(0.15, 0.20, 0.21), 0.2150), BANDS, 0.05)
    assert v["verdict"] == "TIE" and any("f4 guard" in r for r in v["reasons"])


def test_verdict_integer_arithmetic_exact_band_edges():
    """0.3390 - 0.2890 is 0.04999999999999999 in floats; integer ten-thousandths make it exactly the band."""
    assert 0.3390 - 0.2890 != 0.05
    assert F._tt(0.3390) - F._tt(0.2890) == F._tt(0.05) == 500
    c0 = _vin(_p(0.1, 0.1, 0.1), 0.1)
    assert F.verdict(c0, _vin(_p(0.12, 0.1, 0.1), 0.1), BANDS, 0.05)["verdict"] == "TIE"  # brief +0.02 inclusive
    assert F.verdict(c0, _vin(_p(0.1201, 0.1, 0.1), 0.1), BANDS, 0.05)["verdict"] == "LOSE"


def test_log_row_columns_and_basis_in_notes(tmp_path):
    readout = {"timestamp": "2026-10-12T00:00:00+00:00",
               "candidates": {"C0": {"inner": {"pooled": {"brief": 0.1558, "package05": 0.339, "package03": 0.2156,
                                                          "eer": 0.08}}}},
               "verdicts": {}}
    rows = VF.log_rows(readout, "v1", "outputs/v2/fusion/v2_60_readout.json")
    assert set(rows[0]) == set(VF.LOG_COLUMNS)
    assert rows[0]["validation_score"] == rows[0]["validation_score_clean_only"] == "0.3390"
    assert "basis=inner 16,142 nested OOF" in rows[0]["notes"] and rows[0]["benchmark_version"] == "v1"
    log = tmp_path / "log.csv"
    log.write_text(",".join(VF.LOG_COLUMNS) + "\n")
    VF.append_log(log, rows)
    assert len(log.read_text().splitlines()) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,rung\n")
    with pytest.raises(B.BenchmarkError, match="header"):
        VF.append_log(bad, rows)
    assert (REPO / "experiments/log.csv").read_text().splitlines()[0] == ",".join(VF.LOG_COLUMNS)


# ---------------------------------------------------------------------------
# Data cross-check against the independent early look
# ---------------------------------------------------------------------------


@needs_data
def test_inner_numbers_match_independent_early_look():
    cand = json.loads(CAND.read_text())
    bench = VF.VB.load_bench(COLS + ["spectra_aasist"])
    got = {}
    for n in ("C0", "C1", "C3"):
        got[n] = B.reading(bench.pops["inner"].y, bench.score_rule(VF.fixed_rule(n, cand), "inner")["fused"], "package05")
    assert got["C0"] == pytest.approx(0.339045, abs=1e-6)
    assert got["C1"] == pytest.approx(0.452, abs=5e-4)
    assert got["C3"] == pytest.approx(0.358, abs=5e-4)


# ---------------------------------------------------------------------------
# Added after the pre-audit critique
# ---------------------------------------------------------------------------


def test_c2_refuses_non_selectable_population():
    L, y, fold, spectra = world(16)
    for role in ("read_once", "report_only", "in_sample"):
        with pytest.raises(B.BenchmarkError, match="never used for selection"):
            F.run_c2_nested(L, y, fold, spectra, COLS, TARGET,
                            population=B.Population("holdout", role, np.array(["k"]), np.array([0])))
    r = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET,
                        population=B.Population("inner", "selectable", np.array(["k"]), np.array([0])))
    assert r["platt"] is None


def test_overfit_optimism_detects_known_overfit_column():
    """Column 5 tracks the label in folds 0-3 and is reversed in fold 4: for outer
    fold 4 the fit leans on it in-sample, and its edge over C0 collapses out of
    fold, so optimism (advantage_oof - advantage_in) is clearly positive there."""
    rng = np.random.default_rng(17)
    n = 600
    fold = np.array([str(i % 5) for i in range(n)])
    y = (rng.random(n) < 0.5).astype(int)
    sign = np.where(fold == "4", -1.0, 1.0)
    L = np.column_stack([0.5 * y + rng.normal(0, 1, n), 0.3 * y + rng.normal(0, 1, n),
                         0.3 * y + rng.normal(0, 1, n), rng.normal(0, 1, n), 3.0 * sign * y + rng.normal(0, 1, n)])
    spectra = np.full(n, 5.0)
    r = F.run_c2_nested(L, y, fold, spectra, COLS, TARGET)
    f4 = r["folds"]["4"]
    assert f4["advantage_in"]["package05"] < 0  # better than C0 in-sample
    assert f4["optimism"]["package05"] > 0.2
    assert f4["optimism"]["package05"] > max(r["folds"][f]["optimism"]["package05"] for f in "0123")
    assert set(r["optimism_mean"]) == set(B.READINGS)


def test_verdict_none_reading_raises():
    c0 = _vin(_p(0.15, 0.3390, 0.21), 0.1649)
    with pytest.raises(B.BenchmarkError, match="no value"):
        F.verdict(c0, _vin(_p(0.15, None, 0.21), 0.1649), BANDS, 0.05)
    with pytest.raises(B.BenchmarkError, match="no value"):
        F.verdict(c0, _vin(_p(0.15, 0.30, 0.21), None), BANDS, 0.05)


def test_report_refuses_stale_readout(tmp_path):
    if not VF.VB.REGISTRY.exists():
        pytest.skip("no registry")
    blob = F.committed_blob_sha(CAND)
    reg_sha = B.sha256_file(VF.VB.REGISTRY)
    good = {"candidates_blob_sha": blob, "registry_sha256": reg_sha,
            "candidates": {"C0": {"inner": {"pooled": {"brief": 0.1, "package05": 0.2, "package03": 0.1, "eer": 0.05},
                                            "per_fold": {"4": {"package05": 0.1}}}}}, "verdicts": {}}
    f = tmp_path / "r.json"
    f.write_text(json.dumps(good))
    assert "| C0 |" in VF.report(f)
    f.write_text(json.dumps({**good, "registry_sha256": "old"}))
    with pytest.raises(B.BenchmarkError, match="different registry"):
        VF.report(f)
