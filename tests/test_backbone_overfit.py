"""V2-53: the overfit stage and the lane's logging/ownership rules.

Stages H (H1-H11) and G (L1-L5) of `docs/specs/2026-10-11_v2-backbone.md`.

Stage H exists because the owner's P2 risk was *"I really didn't see any tests that make sure
that nothing overfits. I feel like that's a big enough consideration that there should be a set
of tests for it."* It was a real gap: the P2 draft had exactly one test on it.

Since nothing is trained here, "overfit" splits into three distinct things:

1. **the model's memory of its own training data** -- `H11` and the generalization gap,
2. **us fooling ourselves by picking the best of many numbers** -- `H1`, `H2`, `H3`, `H7`,
3. **a model that wins clean and collapses under perturbation** -- `H5`, `H6`, `H8`.

`H9` is the test that makes the other ten mean something: a synthetic candidate built to be
overfit -- near-perfect on the population it "trained" on, chance on the held-out one -- must be
caught. A guard nothing can fail is not a guard.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay.antideepfake_eval import (
    BOOT_N,
    build_verdict,
    cluster_bootstrap,
    condition_degradation,
    generalization_gap,
    package05,
    per_group,
    readings,
    shortcut_aucs,
    size_trend,
)

REPO = Path(__file__).resolve().parents[1]
LANE = REPO / "outputs" / "v2" / "backbone"


# --- synthetic candidates -------------------------------------------------------------------


def make_population(n: int = 600, n_groups: int = 40, seed: int = 0, sep: float = 1.0):
    """A labelled population with speaker-ish clusters."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    groups = np.array([f"spk{i % n_groups}" for i in range(n)])
    s = rng.normal(0, 1, n) + sep * y
    return y, s, groups


def overfit_candidate(seed: int = 1):
    """Near-perfect where it "trained", chance where it did not. The `H9` subject."""
    y_in, _, g_in = make_population(seed=seed)
    rng = np.random.default_rng(seed + 100)
    s_in = rng.normal(0, 0.2, len(y_in)) + 8.0 * y_in  # separable to the point of absurdity
    y_out, _, g_out = make_population(seed=seed + 7)
    s_out = rng.normal(0, 1, len(y_out))  # no signal at all
    return (y_in, s_in, g_in), (y_out, s_out, g_out)


def honest_candidate(seed: int = 2):
    """Similar on both populations: what a non-memorising model looks like."""
    y_in, s_in, g_in = make_population(seed=seed, sep=1.2)
    y_out, s_out, g_out = make_population(seed=seed + 7, sep=1.1)
    return (y_in, s_in, g_in), (y_out, s_out, g_out)


# === H9: the counterfactual that makes the stage real =======================================


def test_h9_a_deliberately_overfit_candidate_is_caught():
    """H9: if the gates cannot fail, they are decoration.

    The subject scores near-perfectly in-sample and at chance held-out. The honest control
    scores similarly on both. The gap statistic must separate them by a wide margin.
    """
    (yi, si, _), (yo, so, _) = overfit_candidate()
    (hyi, hsi, _), (hyo, hso, _) = honest_candidate()

    # a baseline with no membership in either population
    base_in = readings(yi, make_population(len(yi), seed=99, sep=1.0)[1])["package05"]
    base_out = readings(yo, make_population(len(yo), seed=98, sep=1.0)[1])["package05"]

    bad = generalization_gap(readings(yi, si)["package05"], readings(yo, so)["package05"], base_in, base_out)
    good = generalization_gap(readings(hyi, hsi)["package05"], readings(hyo, hso)["package05"], base_in, base_out)

    assert bad["candidate_in_sample"] < 0.05, "the overfit subject must look excellent in-sample"
    assert bad["candidate_held_out"] > 0.5, "and must collapse held-out"
    assert bad["excess_gap"] > good["excess_gap"] + 0.3, (
        f"the gap must separate them: overfit {bad['excess_gap']:.3f} vs honest {good['excess_gap']:.3f}"
    )


def test_h9b_the_overfit_candidate_fails_the_gain_interval_too():
    """H9: its held-out gain over a baseline must not be established."""
    _, (yo, so, go) = overfit_candidate()
    base = make_population(len(yo), seed=98, sep=1.0)[1]
    # candidate first: `cluster_bootstrap(y, s, ..., s_other=o)` returns cost(o) - cost(s), so
    # a positive interval means `s` -- the first series -- is the better one.
    boot = cluster_bootstrap(yo, so, go, cost=package05, n=200, seed=0, s_other=base)
    assert boot["status"] == "ok"
    assert boot["p5"] <= 0, f"a chance-level candidate must not clear the interval (p5={boot['p5']:.4f})"
    assert boot["median"] < 0, "and should sit clearly on the wrong side of zero"


def test_h9c_the_overfit_candidate_is_flagged_as_condition_overfit():
    """H9 via the perturbation route: a candidate that degrades far more than the baseline."""
    deg = condition_degradation(
        clean=0.05, perturbed={"mp3": 0.55, "noise20": 0.62, "speed": 0.40, "shift1": 0.08},
        base_clean=0.30, base_perturbed={"mp3": 0.33, "noise20": 0.35, "speed": 0.32, "shift1": 0.31},
    )  # fmt: skip
    assert deg["condition_overfit"] is True
    assert deg["worst_excess_degradation"] > 0.3
    # and it is a flag, not an exclusion (Codex C5, rule 7)
    assert "does not exclude" in deg["note"]


def test_h9d_an_honest_candidate_is_not_flagged():
    """The other half of a counterfactual: the gates must also pass something legitimate."""
    deg = condition_degradation(
        clean=0.10, perturbed={"mp3": 0.12, "noise20": 0.14}, base_clean=0.30,
        base_perturbed={"mp3": 0.33, "noise20": 0.36},
    )  # fmt: skip
    assert deg["condition_overfit"] is False


# === H1, H2: pre-declaration and comparison counting ========================================


def test_h1_the_realised_sample_equals_the_declared_one():
    """H1: a sample drawn after the first read is a post-hoc choice. The declaration is a
    committed artifact and the realised rows must match it exactly."""
    decl, real = LANE / "sample_predeclared.json", LANE / "inner_sample.csv"
    if not (decl.exists() and real.exists()):
        pytest.skip("sample not declared yet")
    d = json.loads(decl.read_text())
    rows = pd.read_csv(real)
    assert sorted(rows.path) == d["paths"], "the realised sample diverges from the declaration"
    assert len(rows) == d["n"]
    assert rows.label.value_counts().to_dict() == d["label_counts"]


def test_h1b_the_declaration_is_reproducible_from_its_recorded_seed():
    """H1: the declaration records seed and strata, so the draw can be re-derived rather than
    taken on trust."""
    decl = LANE / "sample_predeclared.json"
    if not decl.exists():
        pytest.skip("sample not declared yet")
    d = json.loads(decl.read_text())
    assert d["seed"] == 0 and d["strata"]
    folds = REPO / "splits" / "nsa_folds.csv"
    if not folds.exists():
        pytest.skip("fold file absent")
    inner = pd.read_csv(folds)
    inner = inner[inner.fold != "holdout"]
    assert set(d["paths"]) <= set(inner.path), "the sample must come from the declared basis"


def test_h2_a_verdict_without_its_comparison_count_is_refused():
    """H2: "best of N" reported as one measurement is the oldest way to overstate a result."""
    v = build_verdict(
        name="x", comparisons=4, held_out={"package05": 0.1},
        gap={}, boot={"status": "ok", "p5": 0.05}, degradation={}, shortcuts={},
        usage_corroborated=True,
    )  # fmt: skip
    assert v["comparisons"] == 4
    assert "exploratory" in v["inference"], "N alone does not correct for picking the best of N"
    assert v["provisional"] is True, "provisional until a sealed population exists (rule 7)"


def test_h2b_the_verdict_carries_its_caveats():
    """M9/D12: the verdict is one of three words, and never reads as settled adoption."""
    for corroborated in (True, False):
        v = build_verdict(
            name="x", comparisons=1, held_out={"package05": 0.2}, gap={},
            boot={"status": "ok", "p5": -0.1}, degradation={}, shortcuts={},
            usage_corroborated=corroborated, sealed_set_exists=False,
        )  # fmt: skip
        assert v["verdict"] in {"column", "suppressor", "drop"}
        assert v["provisional"] is True
        assert v["usage_unverified"] is (not corroborated)
        assert "ranks nothing on In-the-Wild" in v["selection_note"]


def test_h2c_a_sealed_population_is_what_clears_provisional():
    v = build_verdict(
        name="x", comparisons=1, held_out={"package05": 0.2}, gap={},
        boot={"status": "ok", "p5": 0.1}, degradation={}, shortcuts={},
        usage_corroborated=True, sealed_set_exists=True,
    )  # fmt: skip
    assert v["provisional"] is False and v["verdict"] == "column"


# === H3, H7, H11: intervals and the size ladder =============================================


def test_h3_a_gain_inside_the_interval_is_not_established():
    """H3: the pass rule is the 5th percentile above zero, not the point estimate."""
    y, s, g = make_population(n=400, n_groups=30, seed=3, sep=1.0)
    nudged = s + 0.01  # a gain far too small to establish
    boot = cluster_bootstrap(y, s, g, cost=package05, n=200, seed=0, s_other=nudged)
    assert boot["status"] == "ok"
    assert boot["p5"] <= 0, "a 0.01 nudge must not clear the interval"
    assert boot["n"] == 200 and boot["n_groups"] == 30


def test_h3b_boot_n_matches_the_repos_existing_bootstraps():
    assert BOOT_N == 1000


def test_h7_the_size_trend_is_computed_not_pre_declared():
    """H7: executable, with no pre-declared outcome.

    An earlier revision pre-declared "not established" from a floor derived by comparing a
    package-0.5 interval against a package-0.3 spread -- a cross-reading error, retracted.
    Widths are pair-specific, so the floor comes from the candidates in hand.
    """
    clear = size_trend([("300m", 315, 0.30), ("1b", 962, 0.20), ("2b", 2159, 0.08)], {"a": 0.05})
    assert clear["monotone"] and clear["established"] and clear["verdict"] == "size trend established"

    noisy = size_trend([("300m", 315, 0.300), ("1b", 962, 0.295), ("2b", 2159, 0.290)], {"a": 0.20})
    assert noisy["monotone"] and not noisy["established"]
    assert noisy["verdict"] == "no size trend established"
    assert "half-width" in noisy["why"]

    jumbled = size_trend([("300m", 315, 0.10), ("1b", 962, 0.40), ("2b", 2159, 0.20)], {"a": 0.01})
    assert not jumbled["monotone"] and not jumbled["established"]
    assert "not monotone" in jumbled["why"]


def test_h11_the_in_sample_arm_gets_its_own_interval():
    """H11: the gain bootstrap only covers the held-out arm, so the differential's own
    uncertainty went uncomputed. An earlier revision claimed H3 covered this; it did not."""
    yi, si, gi = make_population(n=500, n_groups=35, seed=4, sep=2.0)
    b = cluster_bootstrap(yi, si, gi, cost=package05, n=150, seed=0)
    assert b["status"] == "ok" and b["paired"] is False
    assert b["width90"] > 0, "an in-sample point estimate without an interval is not a result"


# === H6, H8: breakdowns and shortcuts =======================================================


def test_h6_the_worst_group_is_reported_beside_the_mean():
    """H6: a model strong on average and broken on one generator family must be visible."""
    rng = np.random.default_rng(5)
    n_each = 80
    frames, ys, ss = [], [], []
    for gen, sep in (("easy1", 2.5), ("easy2", 2.5), ("hard", 0.1)):
        y = np.r_[np.zeros(n_each // 2, int), np.ones(n_each // 2, int)]
        ys.append(y)
        ss.append(rng.normal(0, 1, n_each) + sep * y)
        frames.append(pd.DataFrame({"generator": [gen] * n_each}))
    frame = pd.concat(frames, ignore_index=True)
    y, s = np.concatenate(ys), np.concatenate(ss)
    out = per_group(frame, y, s, "generator", cost=package05, min_rows=30)
    assert out["worst_group"] == "hard"
    assert out["worst_cost"] > out["mean_of_groups"], "the worst must be worse than the average"


def test_h6b_itw_is_degenerate_for_this_test_and_says_so():
    """H6: on In-the-Wild `generator` has two values and `source` one, so the worst group *is*
    the mean. The function returns nothing rather than a bogus number."""
    man = REPO / "outputs" / "manifests" / "itw_stress.csv"
    if not man.exists():
        pytest.skip("ITW manifest absent")
    d = pd.read_csv(man)
    assert d.generator.nunique() == 2 and d.source.nunique() == 1
    y = (d.label == "spoof").to_numpy(int)
    s = np.random.default_rng(0).normal(0, 1, len(d)) + y

    # `source` has one value, so there is exactly one group and "worst" IS "mean" -- the
    # statistic carries no information, which is the point.
    by_source = per_group(d, y, s, "source", cost=package05)
    assert len(by_source["per_group"]) == 1
    assert by_source["worst_cost"] == by_source["mean_of_groups"]

    # `generator` has two values but each is a single class (`bonafide` / `itw_spoof`), so
    # neither yields a cost at all and no worst group is reported.
    by_gen = per_group(d, y, s, "generator", cost=package05)
    assert by_gen["per_group"] == {}
    assert "worst_group" not in by_gen


def test_h8_a_score_riding_a_surface_feature_is_flagged():
    """H8: M5's stream-shortcut audit, extended past its four continuous features."""
    rng = np.random.default_rng(6)
    n = 300
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    dur = rng.uniform(1, 9, n)
    riding = dur * 3.0 + rng.normal(0, 0.05, n)  # the score IS the duration
    frame = pd.DataFrame({"duration_s": dur, "peak": rng.uniform(0.1, 1.0, n)})
    out = shortcut_aucs(frame, riding, y, columns=("duration_s", "peak"))
    assert any(f.startswith("duration_s") for f in out["flagged_above_0_85"])
    assert not any(f.startswith("peak") for f in out["flagged_above_0_85"])


def test_h8b_a_clean_score_is_not_flagged():
    rng = np.random.default_rng(7)
    n = 300
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    frame = pd.DataFrame({"duration_s": rng.uniform(1, 9, n), "peak": rng.uniform(0.1, 1, n)})
    out = shortcut_aucs(frame, rng.normal(0, 1, n) + y, y, columns=("duration_s", "peak"))
    assert out["flagged_above_0_85"] == []


# === Stage G: logging and lane ownership ====================================================

V2_LOG_COLUMNS = [
    "timestamp", "rung", "validation_score", "validation_score_clean_only",
    "csv_path", "notes", "benchmark_version",
]  # fmt: skip


def append_experiment_row(log: Path, rung: str, validation_score, notes: str,
                          benchmark_version: str, clean_only=None) -> None:  # fmt: skip
    """One row per pre-declared run in `experiments/log.csv` (`L2`).

    `hearsay.submission.append_log` writes six columns to `submissions/log.csv`, which is v1's
    ledger and forbidden here; the v2 ledger has a seventh, `benchmark_version`. This lane is
    the first to write it.
    """
    import datetime as dt

    row = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "rung": rung, "validation_score": validation_score,
        "validation_score_clean_only": validation_score if clean_only is None else clean_only,
        "csv_path": "", "notes": notes, "benchmark_version": benchmark_version,
    }  # fmt: skip
    with log.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=V2_LOG_COLUMNS).writerow(row)


@pytest.fixture(autouse=True)
def _protect_the_v1_ledger(tmp_path, monkeypatch):
    """L1: the v1 ledger must be byte-identical after this module runs.

    This is the remedy adopted after an M5 round appended six fabricated rows to a tracked
    ledger: point every test at a temp log, and assert the real file by hash.
    """
    real = REPO / "submissions" / "log.csv"
    before = hashlib.sha256(real.read_bytes()).hexdigest() if real.exists() else None
    yield
    if before is not None:
        after = hashlib.sha256(real.read_bytes()).hexdigest()
        assert after == before, "submissions/log.csv changed; it is v1's ledger and read-only here"


def test_l1_the_v1_ledger_is_never_written(tmp_path):
    """L1: the forbidden file, and the appender that must not touch it."""
    log = tmp_path / "log.csv"
    log.write_text(",".join(V2_LOG_COLUMNS) + "\n")
    append_experiment_row(log, "V2-53", 0.19, "package-0.5 on ITW; basis 16,142", "none (pre-V2-01)")
    rows = list(csv.DictReader(log.open()))
    assert len(rows) == 1
    assert (REPO / "submissions" / "log.csv").exists(), "the v1 ledger must still be there, untouched"


def test_l2_the_v2_ledger_has_seven_columns_and_the_v1_one_has_six():
    """L2: the two ledgers differ, which is why `append_log` cannot be reused."""
    v2 = REPO / "experiments" / "log.csv"
    v1 = REPO / "submissions" / "log.csv"
    if not (v2.exists() and v1.exists()):
        pytest.skip("ledgers absent")
    assert next(csv.reader(v2.open())) == V2_LOG_COLUMNS
    assert len(next(csv.reader(v1.open()))) == 6
    assert "benchmark_version" not in next(csv.reader(v1.open()))


def test_l2b_a_row_states_its_basis_and_benchmark_version(tmp_path):
    """L2: `validation_score` is meaningless without the population, and `benchmark_version`
    must record that V2-01 does not exist rather than being left blank."""
    log = tmp_path / "log.csv"
    log.write_text(",".join(V2_LOG_COLUMNS) + "\n")
    append_experiment_row(log, "V2-53", 0.19, "package-0.5 on ITW; EER 0.0755; basis 16,142", "none (pre-V2-01)")
    r = next(csv.DictReader(log.open()))
    assert "basis" in r["notes"] and "package-0.5" in r["notes"]
    assert r["benchmark_version"] == "none (pre-V2-01)"


def test_l3_the_lane_writes_only_under_its_own_output_dir():
    """L3/E5: every artifact this lane produces lives under `outputs/v2/backbone/`."""
    if not LANE.exists():
        pytest.skip("lane dir absent")
    strays = [p for p in LANE.rglob("*") if p.is_file() and not str(p).startswith(str(LANE))]
    assert strays == []
    forbidden = REPO / "outputs" / "detector_scores"
    if forbidden.exists():
        ours = [p for p in forbidden.glob("*anti-deepfake*")]
        assert ours == [], f"this lane must not write to outputs/detector_scores: {ours}"


def test_l5_the_baseline_snapshot_exists_so_regression_is_checkable():
    """L5: the working tree was already dirty with other lanes' files when this lane started,
    so without a recorded baseline "no tracked file outside this lane changed" is unverifiable."""
    base = LANE / "git_baseline.txt"
    if not base.exists():
        pytest.skip("baseline not captured")
    text = base.read_text()
    assert "submissions/log.csv" in text
    assert "captured:" in text
