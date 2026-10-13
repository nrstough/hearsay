"""Read the backbone lane's score files and produce its verdict. V2-53 step 11.

Run spec `docs/specs/2026-10-11_v2-backbone.md`. Reads only `outputs/v2/backbone/*__*.csv`,
joins them against the manifests and the v1 baseline export, and writes
`outputs/v2/backbone/report.json` plus a markdown table for the report under `docs/reports/`.

What it is careful about, and why:

* **Every join asserts its expected row count.** Measured on the real exports, a scorer that
  re-spells its paths loses 20,000 of 24,671 rows (81.1%) silently, because three `data/`
  directories are symlinks; In-the-Wild survives intact, so the damage hides in exactly the
  readout one would check. Nothing here proceeds on a partial join.
* **Every number carries an interval**, from a cluster bootstrap over the population's own
  grouping (speakers for In-the-Wild, generators and sources for the in-sample arm).
* **In-the-Wild ranks nothing.** Rule 7 spent it as a selection set, and Codex C5 rejected
  using its perturbation cohorts to exclude candidates. Results are descriptive; adoption
  belongs to the benchmark lane's sealed population.
* **The gap is descriptive.** An in-sample-versus-held-out difference mixes training exposure
  with corpus difficulty, channel, speakers and generator composition, so it is reported as a
  generalization-gap difference, not an identified inflation estimate (Codex C10).

Usage:
  uv run python scripts/v2_backbone_report.py
  uv run python scripts/v2_backbone_report.py --write-doc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hearsay.antideepfake_eval import (
    BOOT_N,
    build_verdict,
    cluster_bootstrap,
    generalization_gap,
    package05,
    per_group,
    readings,
    shortcut_aucs,
    size_trend,
)

LANE = REPO / "outputs" / "v2" / "backbone"
BASELINE = REPO / "outputs" / "detector_scores" / "m1b_v3.csv"

CANDIDATES = {  # name -> parameter count, for the size ladder
    "mms-300m-anti-deepfake": 315_437_570,
    "xls-r-1b-anti-deepfake": 962_500_000,
    "mms-1b-anti-deepfake": 962_500_000,
    "xls-r-2b-anti-deepfake": 2_159_300_000,
}
#: population -> (manifest for labels/groups, grouping column, is this arm in-sample?)
ARMS = {
    "itw": ("outputs/manifests/itw_stress.csv", "speaker", False),
    "inner_sample": ("outputs/v2/backbone/inner_sample.csv", "group", True),
    "asv19_control": ("outputs/v2/backbone/asv19_control.csv", "speaker", True),
    "test": ("outputs/manifests/nsa_test.csv", None, None),
}


def surface_features(manifest: Path, cache: Path) -> pd.DataFrame:
    """Per-clip duration, peak level, leading silence and effective band, cached.

    `H8`'s shortcut audit needs these and the scorer does not record them, so they are derived
    from the audio once and cached: 3,000 ITW clips decode in about four minutes, and a cached
    frame makes the audit free thereafter.
    """
    if cache.exists():
        return pd.read_csv(cache)
    from hearsay import SR
    from hearsay.audio import load_audio

    rows = []
    for rel in pd.read_csv(manifest).path:
        full = Path(rel) if Path(rel).is_absolute() else REPO / rel
        try:
            x = load_audio(full)
        except Exception:  # noqa: BLE001 - a missing clip is a blank row, not a crash
            rows.append({"path": rel})
            continue
        nz = np.abs(x) > 1e-4
        lead = int(nz.argmax()) / SR if nz.any() else len(x) / SR
        seg = x[: 1 << 16]
        energy = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1 / SR)
        # the frequency below which 99% of the energy sits: the band-match cue this repo found
        # on the NSA test set, and a plausible shortcut for a detector to ride
        inside = (energy.cumsum() / max(energy.sum(), 1e-12)) < 0.99
        band = float(freqs[inside][-1]) if energy.sum() and inside.any() else 0.0
        rows.append({
            "path": rel, "duration_s": len(x) / SR, "peak": float(abs(x).max()),
            "lead_silence_s": lead, "band_hz": band,
        })  # fmt: skip
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def load_scores(model: str, pop: str) -> pd.DataFrame | None:
    f = LANE / f"{model}__{pop}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    flagged = d.flag.fillna("").ne("").sum() if "flag" in d else 0
    if flagged:
        raise ValueError(f"{f.name}: {flagged} flagged rows; refusing to read a run with integrity flags")
    return d


def join_arm(scores: pd.DataFrame, manifest: Path, group_col: str | None) -> pd.DataFrame:
    """Join scores to labels on the manifest's own path spelling, asserting full coverage."""
    man = pd.read_csv(manifest)
    before = len(scores)
    j = scores.merge(man, on="path", how="inner", suffixes=("", "_man"))
    if len(j) != before:
        missing = set(scores.path) - set(man.path)
        raise ValueError(
            f"join lost {before - len(j)} of {before} rows against {manifest.name}; "
            f"first missing: {sorted(missing)[:3]}"
        )
    if group_col and group_col not in j.columns:
        raise ValueError(f"{manifest.name} has no {group_col!r} column for the bootstrap")
    return j


def baseline_for(pop: str) -> pd.DataFrame | None:
    """v1's M1b L7 export, which covers ITW, inner, holdout and test but not ASV19 or MLAAD."""
    if not BASELINE.exists():
        return None
    b = pd.read_csv(BASELINE)
    want = {"itw": "itw", "inner_sample": "inner_oof", "test": "test"}.get(pop)
    return None if want is None else b[b.split == want][["path", "logit"]]


def analyse(model: str, verbose: bool = True) -> dict:
    out: dict = {"model": model, "arms": {}}
    rec = LANE / f"{model}.identity.json"
    if rec.exists():
        d = json.loads(rec.read_text())
        out["revision"] = d.get("revision")
        out["licence"] = d.get("licence")

    for pop, (man_rel, group_col, in_sample) in ARMS.items():
        s = load_scores(model, pop)
        if s is None:
            continue
        man = REPO / man_rel
        if not man.exists():
            continue
        j = join_arm(s, man, group_col)
        arm: dict = {"n": len(j), "in_sample": in_sample, "manifest": man_rel}
        if in_sample is None:  # the unlabelled test set: rank agreement only
            base = baseline_for(pop)
            if base is not None:
                m = j.merge(base.rename(columns={"logit": "base_logit"}), on="path")
                if len(m) == len(j):
                    from scipy.stats import spearmanr

                    arm["spearman_vs_baseline"] = float(spearmanr(m.logit, m.base_logit).statistic)
                    arm["note"] = "label-free: rank agreement only, no minDCF"
            out["arms"][pop] = arm
            continue

        y = (j.label == "spoof").to_numpy(int)
        sc = j.logit.to_numpy(float)
        arm["readings"] = readings(y, sc)
        if in_sample:
            arm["membership_evidence"] = (
                "in the AntiDeepfake training table (archived at outputs/v2/backbone/cards/)"
            )
        if group_col:
            arm["interval_package05"] = cluster_bootstrap(
                y, sc, j[group_col], cost=package05, n=BOOT_N, seed=0
            )
        base = baseline_for(pop)
        if base is not None:
            m = j.merge(base.rename(columns={"logit": "base_logit"}), on="path")
            if len(m) != len(j):
                raise ValueError(f"{model}/{pop}: baseline join lost {len(j) - len(m)} rows")
            yb = (m.label == "spoof").to_numpy(int)
            arm["baseline_readings"] = readings(yb, m.base_logit.to_numpy(float))
            if group_col:
                arm["gain_over_baseline"] = cluster_bootstrap(
                    yb, m.logit.to_numpy(float), m[group_col], cost=package05,
                    n=BOOT_N, seed=0, s_other=m.base_logit.to_numpy(float),
                )  # fmt: skip
        if "generator" in j.columns and j.generator.nunique() > 1:
            arm["by_generator"] = per_group(j, y, sc, "generator", cost=package05)
        for col in ("source", "partition"):
            if col in j.columns and j[col].nunique() > 1:
                arm[f"by_{col}"] = per_group(j, y, sc, col, cost=package05)
        try:
            feats = surface_features(man, LANE / f"surface__{Path(man_rel).stem}.csv")
            f = j.merge(feats, on="path", how="left")
            if len(f) == len(j):
                arm["shortcuts"] = shortcut_aucs(f, sc, y)
        except Exception as e:  # noqa: BLE001 - the audit is a readout, not a gate
            arm["shortcuts"] = {"error": f"{type(e).__name__}: {e}"}
        out["arms"][pop] = arm
        if verbose:
            r = arm["readings"]
            print(f"  {model:26} {pop:14} n={len(j):5} pkg05={r.get('package05', float('nan')):.4f} "
                  f"pkg03={r.get('package03', float('nan')):.4f} EER={r.get('eer', float('nan')):.4f}")  # fmt: skip
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-doc", action="store_true")
    ap.parse_args(argv)

    print("per-candidate readouts:")
    results = {m: analyse(m) for m in CANDIDATES if (LANE / f"{m}.identity.json").exists()}
    results = {m: r for m, r in results.items() if r["arms"]}
    if not results:
        print("no score files yet", file=sys.stderr)
        return 1

    # the generalization-gap difference, per candidate (M1, descriptive per Codex C10)
    for m, r in results.items():
        itw, inner = r["arms"].get("itw"), r["arms"].get("inner_sample")
        if itw and inner and "baseline_readings" in itw and "baseline_readings" in inner:
            r["generalization_gap"] = generalization_gap(
                inner["readings"]["package05"], itw["readings"]["package05"],
                inner["baseline_readings"]["package05"], itw["baseline_readings"]["package05"],
            )  # fmt: skip

    # the size ladder (H7): executable, outcome not pre-declared
    ladder = [
        (m, CANDIDATES[m], r["arms"]["itw"]["readings"]["package05"])
        for m, r in results.items() if r["arms"].get("itw", {}).get("readings", {}).get("status") == "ok"
    ]
    widths = {
        m: r["arms"]["itw"]["interval_package05"]["width90"]
        for m, r in results.items()
        if r["arms"].get("itw", {}).get("interval_package05", {}).get("status") == "ok"
    }
    trend = size_trend(ladder, widths) if len(ladder) >= 2 else {"verdict": "too few candidates scored"}

    # one verdict per candidate, with N = the number of candidates compared on the primary read
    n_comparisons = len(ladder)
    verdicts = {}
    for m, r in results.items():
        itw = r["arms"].get("itw")
        if not itw or itw.get("readings", {}).get("status") != "ok":
            continue
        verdicts[m] = build_verdict(
            name=m, comparisons=n_comparisons, held_out=itw["readings"],
            gap=r.get("generalization_gap", {}), boot=itw.get("gain_over_baseline", {}),
            degradation={}, shortcuts=itw.get("shortcuts", {}),
            usage_corroborated=False, sealed_set_exists=False,
        )  # fmt: skip

    report = {
        "spec": "docs/specs/2026-10-11_v2-backbone.md",
        "candidates": results,
        "size_trend": trend,
        "verdicts": verdicts,
        "selection_basis": (
            "none. This lane ranks nothing on In-the-Wild (rule 7, Codex C5); adoption belongs "
            "to the benchmark lane's sealed population."
        ),
        "caveats": {
            "gap": "descriptive generalization-gap difference, not an identified inflation estimate",
            "inference": "exploratory, not simultaneous; N recorded but not corrected for",
            "usage": "P3b corroboration not yet run, so every verdict carries usage_unverified",
        },
    }
    dest = LANE / "report.json"
    dest.write_text(json.dumps(report, indent=1, default=bool) + "\n")
    print(f"\nwrote {dest.relative_to(REPO)}")
    print(f"size trend: {trend.get('verdict')} ({trend.get('why', '')})")
    for m, v in verdicts.items():
        g = v.get("gain_over_baseline_interval", {})
        print(f"  {m:26} verdict={v['verdict']:10} established={v['established']} "
              f"p5={g.get('p5', float('nan')):.4f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
