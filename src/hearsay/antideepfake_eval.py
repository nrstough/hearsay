"""Readouts, intervals and overfit gates for the V2-53 backbone lane.

Run spec `docs/specs/2026-10-11_v2-backbone.md`, stages F (M1-M11) and H (H1-H11).

Why this exists rather than reusing what the repo has:

* **`hearsay.metrics.report()` does not give package-0.5 usably.** Its `min_dcf_pi05` key is
  `min_cost(y, s, 0.5)` with the *default* costs, i.e. 4·P_FA + P_miss -- measured 0.5448 where
  package-0.5 is 0.5229. The key that *is* package-0.5 is `sponsor_code_flipped`, but it is
  rounded to 4 dp (0.5229 vs 0.522857142857143), which is unusable for a bootstrap or a
  differential. So the three readings are computed by calling `min_cost` directly.
* **Package-0.3 has no helper anywhere in the repo.** Verified equal to the winning team's
  stated FPR + 1.714·FNR formula to 5.6e-17, so there is no convention gap with their numbers.
* **`by_group` hardcodes the brief costs** (`src/hearsay/metrics.py:113`); `pi` is its only knob.
  A per-generator table at package-0.5 needs its own implementation.
* **The existing speaker-cluster bootstrap** (`scripts/bakeoff_layers.py:219` `boot_gain`) has
  the right shape -- injectable cost, 1,000 resamples -- but lacks two guards that
  `scripts/fuse_sweep_v3.py:239` has and rounds to 4 dp. Both guards are ported here.

`min_cost` **clamps at 1.0**: an inverted or random detector reads exactly 1.0. That means "no
better than the constant decision", not "break-even", and nothing here interprets it otherwise.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from hearsay.metrics import eer, min_cost

#: Resamples for every interval. Matches `BOOT_N` in the repo's existing bootstraps.
BOOT_N = 1000
BOOT_SEED = 0


def package05(y, s) -> float:
    """NSA's shipped minDCF at Pspoof 0.5: **P_FA + 4·P_miss**. The selection metric (rule 5).

    Identical to `sponsor_min_dcf(y, s, flip=True)`; measured agreement 1.11e-16.
    """
    return float(min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0))


def package03(y, s) -> float:
    """The package at Pspoof 0.3: **P_FA + 1.714·P_miss**. What other teams say NSA scored with."""
    return float(min_cost(y, s, 0.3, c_fa=1.0, c_miss=4.0))


def brief(y, s) -> float:
    """v1's reading: **9.33·P_FA + P_miss**. Every v1 report and research pilot uses it."""
    return float(min_cost(y, s))


COSTS = {"package05": package05, "package03": package03, "brief": brief}


def readings(y, s) -> dict:
    """All three readings plus EER. Rule 5 requires EER beside every reading (`M11`)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    finite = np.isfinite(s)
    if finite.sum() == 0 or np.unique(y[finite]).size < 2:
        return {"status": "degenerate", "n_used": int(finite.sum())}
    y, s = y[finite], s[finite]
    return {
        "status": "ok",
        "n_used": int(finite.sum()),
        "n_dropped": int((~finite).sum()),
        **{name: fn(y, s) for name, fn in COSTS.items()},
        "eer": float(eer(y, s)),
    }


def _members(groups) -> list[np.ndarray]:
    """Row indices per cluster, refusing missing group labels.

    `boot_gain` lacked this: a NaN or blank speaker silently became its own cluster.
    """
    g = pd.Series(groups)
    if g.isna().any() or (g.astype(str).str.strip() == "").any():
        raise ValueError("bootstrap: missing speaker/group values")
    ids = pd.factorize(g)[0]
    return [np.flatnonzero(ids == k) for k in range(ids.max() + 1)]


def cluster_bootstrap(
    y, s, groups, cost=package05, n: int = BOOT_N, seed: int = BOOT_SEED, s_other=None,
) -> dict:  # fmt: skip
    """Cluster bootstrap of one cost, or of a **paired** difference when `s_other` is given.

    Resamples whole clusters, so dependence between clips of the same speaker or generator is
    carried. Single-class draws are skipped **and counted**: `roc_curve` returns NaN rather than
    raising on them, which would silently poison the percentile.

    With `s_other`, returns the interval on `cost(s_other) - cost(s)`; positive means `s` is
    better. Candidate and reference see **identical resamples** (`H4`).
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    members = _members(groups)
    rng = np.random.default_rng(seed)
    draws, skipped = [], 0
    other = None if s_other is None else np.asarray(s_other, dtype=np.float64)
    for _ in range(n):
        pick = rng.choice(len(members), len(members), replace=True)
        rows = np.concatenate([members[k] for k in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            skipped += 1
            continue
        if other is None:
            draws.append(cost(yy, s[rows]))
        else:
            draws.append(cost(yy, other[rows]) - cost(yy, s[rows]))
    if not draws:
        return {"status": "no_valid_draws", "n": n, "skipped_single_class": skipped}
    d = np.array(draws, dtype=np.float64)
    p5, p50, p95 = (float(v) for v in np.percentile(d, [5, 50, 95]))
    return {
        "status": "ok", "p5": p5, "median": p50, "p95": p95, "width90": p95 - p5,
        "n": n, "skipped_single_class": skipped, "n_groups": len(members),
        "paired": other is not None,
    }  # fmt: skip


def per_group(frame: pd.DataFrame, y, s, column: str, cost=package05, min_rows: int = 30) -> dict:
    """A per-generator / per-source table at **any** cost, with the worst group beside the mean.

    `hearsay.metrics.by_group` cannot do this: it hardcodes the brief costs. It also needs a
    fresh RangeIndex, a trap this avoids by taking positional arrays.

    Groups below `min_rows` are reported but excluded from the worst-group statistic -- a
    max over noisy small-sample estimates is biased upward, and more so the smaller they are.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    vals, thin = {}, {}
    for name, idx in frame.reset_index(drop=True).groupby(column).groups.items():
        rows = np.asarray(idx)
        yy = y[rows]
        if np.unique(yy).size < 2:
            continue  # a single-class group has no cost
        v = float(cost(yy, s[rows]))
        (vals if len(rows) >= min_rows else thin)[str(name)] = {"cost": v, "n": len(rows)}
    out = {"per_group": {**vals, **thin}, "excluded_below_min_rows": sorted(thin)}
    if vals:
        worst = max(vals, key=lambda k: vals[k]["cost"])
        out["worst_group"] = worst
        out["worst_cost"] = vals[worst]["cost"]
        out["mean_of_groups"] = float(np.mean([v["cost"] for v in vals.values()]))
    return out


def generalization_gap(cand_in: float, cand_out: float, base_in: float, base_out: float) -> dict:
    """The owner's P1 risk as a number: how much more a candidate's score drops going from a
    population it was trained on to one it was not, compared with an untrained baseline.

    **Descriptive, not causal** (Codex C10). An in-sample-versus-In-the-Wild difference mixes
    training exposure with corpus difficulty, channel, speakers and generator composition;
    subtracting the baseline's gap does not isolate memorisation. Reported as a
    generalization-gap difference, and the report says so.
    """
    cand_gap = cand_out - cand_in
    base_gap = base_out - base_in
    return {
        "candidate_in_sample": cand_in, "candidate_held_out": cand_out, "candidate_gap": cand_gap,
        "baseline_in_sample": base_in, "baseline_held_out": base_out, "baseline_gap": base_gap,
        "excess_gap": cand_gap - base_gap,
        "interpretation": "descriptive generalization-gap difference, not an identified inflation estimate",
    }  # fmt: skip


def size_trend(ladder: list[tuple[str, int, float]], widths: dict[str, float]) -> dict:
    """Does cost improve with parameter count, **beyond** the intervals? (`H7`)

    Executable, with **no pre-declared outcome**. An earlier revision pre-declared "not
    established" from a floor derived by comparing a package-0.5 interval against a package-0.3
    spread -- a cross-reading error, retracted. Paired widths are pair-specific (measured 0.158
    to 0.336 at package-0.5 across three real pairs), so the floor comes from the candidates
    actually in hand.

    **Ties in parameter count are grouped, not ordered.** A first version sorted every candidate
    into a sequence, so two models of *identical* size (XLS-R-1B and MMS-1B are both 962.5M)
    were compared as consecutive rungs and the ladder read "not monotone" purely from their
    arbitrary order. That is an artifact of sorting a tie, not a result. Same-size candidates
    now form one rung, the trend runs over rung means, and the spread *within* a rung is
    reported separately -- because a within-rung spread wider than the interval is itself a
    finding: it says architecture matters more than size at that scale.
    """
    rungs: dict[int, list[tuple[str, float]]] = {}
    for name, params, cost in ladder:
        rungs.setdefault(params, []).append((name, cost))
    sizes = sorted(rungs)
    means = [float(np.mean([c for _, c in rungs[s]])) for s in sizes]
    floor = max(widths.values()) / 2 if widths else float("inf")

    improving = all(b <= a for a, b in itertools.pairwise(means))
    span = max(means) - min(means) if means else 0.0
    established = improving and span > floor and len(sizes) >= 2

    within = {}
    for s in sizes:
        if len(rungs[s]) > 1:
            costs = [c for _, c in rungs[s]]
            within[s] = {
                "members": {n: c for n, c in rungs[s]},
                "spread": max(costs) - min(costs),
                "exceeds_interval": (max(costs) - min(costs)) > floor,
            }

    if not improving:
        why = "rung means are not monotone in parameter count"
    elif span <= floor:
        why = f"span {span:.4f} does not clear the largest half-width {floor:.4f}"
    else:
        why = f"span {span:.4f} clears the largest half-width {floor:.4f}"

    return {
        "rungs": {str(s): {n: c for n, c in rungs[s]} for s in sizes},
        "rung_params": sizes, "rung_means": means,
        "monotone": improving, "span": span, "largest_half_width": floor,
        "established": established,
        "verdict": "size trend established" if established else "no size trend established",
        "why": why,
        "within_rung_spread": within,
        "note": (
            "same-size candidates form one rung; a within-rung spread wider than the interval "
            "means architecture matters more than size at that scale"
        ),
    }  # fmt: skip


def shortcut_aucs(frame: pd.DataFrame, s, y, columns=("duration_s", "peak", "lead_silence_s", "band_hz")) -> dict:
    """Within-class rank correlation of the score against cheap surface features (`H8`).

    Measured on the realised scoring stream, the way M5's stream-shortcut audit was. A value
    above 0.85 means the score is partly riding the feature rather than the audio.
    """
    from scipy.stats import spearmanr

    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y).astype(int)
    out, flagged = {}, []
    f = frame.reset_index(drop=True)
    for col in columns:
        if col not in f.columns:
            continue
        v = pd.to_numeric(f[col], errors="coerce").to_numpy(dtype=np.float64)
        per_class = {}
        for cls, label in ((0, "bonafide"), (1, "spoof")):
            m = np.isfinite(v) & np.isfinite(s) & (y == cls)
            if m.sum() < 3 or np.ptp(v[m]) == 0 or np.ptp(s[m]) == 0:
                per_class[label] = None
                continue
            r = float(spearmanr(v[m], s[m]).statistic)
            auc = 0.5 + abs(r) / 2  # a rank correlation read as a one-feature AUC
            per_class[label] = {"spearman": r, "auc_equivalent": auc, "n": int(m.sum())}
            if auc > 0.85:
                flagged.append(f"{col}:{label}")
        out[col] = per_class
    return {"within_class": out, "flagged_above_0_85": sorted(set(flagged))}


def condition_degradation(clean: float, perturbed: dict[str, float], base_clean: float,
                          base_perturbed: dict[str, float], margin: float = 0.030) -> dict:  # fmt: skip
    """Clean-to-perturbed degradation against the baseline's (`H5`).

    **A reported flag, never an exclusion** (Codex C5): excluding a candidate on In-the-Wild
    perturbations is selection on In-the-Wild, which rule 7 bars however it is phrased.
    """
    rows = {}
    for kind, v in sorted(perturbed.items()):
        b = base_perturbed.get(kind)
        rows[kind] = {
            "candidate": v, "candidate_degradation": v - clean,
            "baseline": b, "baseline_degradation": None if b is None else b - base_clean,
            "excess": None if b is None else (v - clean) - (b - base_clean),
        }  # fmt: skip
    excess = [r["excess"] for r in rows.values() if r["excess"] is not None]
    worst = max(excess) if excess else None
    return {
        "margin": margin, "per_kind": rows, "worst_excess_degradation": worst,
        "condition_overfit": bool(worst is not None and worst > margin),
        "note": "descriptive flag; does not exclude the candidate (Codex C5, rule 7)",
    }  # fmt: skip


def refuse_degenerate(y, s) -> None:
    """A constant-scoring or worse-than-chance candidate is refused, not reported with its
    flattering in-sample number (`H10`)."""
    s = np.asarray(s, dtype=np.float64)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        raise ValueError("every score is non-finite")
    if np.ptp(finite) == 0:
        raise ValueError(f"constant score {finite[0]!r}; refusing to report a degenerate candidate")
    from sklearn.metrics import roc_auc_score

    m = np.isfinite(s)
    auc = float(roc_auc_score(np.asarray(y).astype(int)[m], s[m]))
    if auc <= 0.5:
        raise ValueError(f"AUC {auc:.4f} <= 0.5 on the held-out population; refusing to report it")


def build_verdict(
    name: str, comparisons: int, held_out: dict, gap: dict, boot: dict,
    degradation: dict, shortcuts: dict, usage_corroborated: bool, sealed_set_exists: bool = False,
) -> dict:  # fmt: skip
    """column / suppressor / drop, with everything a reader needs to discount it.

    `provisional` is `True` until a sealed population outside the AntiDeepfake training list
    exists (rule 7, `D12`, `M9`). `comparisons` is the N behind the choice (`H2`), and inference
    is **exploratory, not simultaneous** -- recording N does not correct for picking the
    strongest of several intervals, and the report says so.
    """
    improves = boot.get("status") == "ok" and boot.get("p5", -1) > 0
    verdict = "column" if improves else ("suppressor" if held_out.get("package05", 1.0) < 1.0 else "drop")
    return {
        "candidate": name,
        "verdict": verdict,
        "provisional": not sealed_set_exists,
        "usage_unverified": not usage_corroborated,
        "comparisons": comparisons,
        "inference": "exploratory, not simultaneous; N is recorded but not corrected for",
        "held_out_readings": held_out,
        "gain_over_baseline_interval": boot,
        "established": improves,
        "generalization_gap": gap,
        "condition": degradation,
        "shortcuts": shortcuts,
        "selection_note": (
            "this lane ranks nothing on In-the-Wild; adoption belongs to the benchmark lane's "
            "sealed population (rule 7, Codex C5)"
        ),
    }
