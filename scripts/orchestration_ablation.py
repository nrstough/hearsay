"""Orchestration ablation: the shipped router on and off, on the labeled outer holdout and the
eval-only In-the-Wild set, plus the unlabeled test set's share above 0.5.

No model runs. It re-fuses the exported logits (outputs/detector_scores/, the In-the-Wild
companions) exactly as the fusion constants file prescribes (fusion_v1: alpha over M1b + handcrafted; fusion_v2: weights over M1b + handcrafted + M5), then switches each routing
rule off in turn. Two rules can change a file's score; every other detector only routes or
explains:

  E   M3 (Spectra-AASIST) as false-alarm suppression: when its margin is below -3 ("strongly
      bona fide") and the base rank is above 0.5, the base rank is halved. M3 never raises a score.
  G   the non-speech gate: a file with no speech to judge (speech_gate.is_speech == 0) is pinned
      below every determinate score (hearsay.detectors.speech_gate.apply_default_answer).

Configurations
  router_on        base rank blend + E + G        (shipped: E_on_A_alpha0.2 with the gate policy)
  no_gate          base + E
  no_suppression   base + G
  router_off       base only: every fused detector always runs, no gate, no M3 rule, no block
  fuse_everything  rank mean of M1b, handcrafted and M3, no rules: the equal-weight fusion the
                   consult rejected (In-the-Wild misses doubled)

Readouts per split: normalized minDCF under the brief's cost (9.33 P_FA + P_miss at the 0.3
prior) and under the sponsor code's cost weighting, the number of files each rule touched, and
how many files the evidence-only detectors flagged without changing any score.

Usage: uv run python scripts/orchestration_ablation.py [--constants models/fusion_v2/constants.json --out outputs/fusion/orchestration_ablation_v2]
Writes outputs/fusion/orchestration_ablation.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from fuse_sweep import averse, brief, load, norm

from hearsay.detectors.speech_gate import BLOCK_TOP, apply_default_answer

S = REPO / "outputs" / "detector_scores"
SPLITS = ("holdout", "itw", "test")


def rank_of(logit: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.searchsorted(ref, logit, side="left") / len(ref)


def gate_flags() -> pd.Series:
    """is_speech per path: the export for fold and test rows, the inventory for In-the-Wild."""
    g = pd.read_csv(S / "speech_gate_results.csv")
    g["is_speech"] = [json.loads(f).get("is_speech", 1.0) for f in g.features]
    itw = pd.read_csv(REPO / "outputs" / "inventory" / "itw_speech_gate.csv")
    both = pd.concat([g[["path", "is_speech"]], itw[["path", "is_speech"]]], ignore_index=True)
    both["path"] = both.path.map(norm)
    return both.drop_duplicates("path").set_index("path").is_speech.astype(float)


def evidence_flags(paths: pd.Index) -> dict[str, int]:
    """How many of these rows each evidence-only detector flagged (fold and test rows only)."""
    out: dict[str, int] = {}
    for name, key in (("enf", "enf_present"), ("splice", "n_seams"), ("speaker_drift", "drift")):
        p = S / f"{name}_results.csv"
        if not p.exists():
            out[name] = -1
            continue
        r = pd.read_csv(p)
        r["path"] = r.path.map(norm)
        r = r.set_index("path").reindex(paths)
        vals = [json.loads(f).get(key, 0.0) if isinstance(f, str) else 0.0 for f in r.features]
        out[name] = int(sum(1 for v in vals if v and v > 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constants", type=Path, default=REPO / "models" / "fusion_v1" / "constants.json")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "fusion" / "orchestration_ablation")
    args = ap.parse_args()
    c = json.loads(args.constants.read_text())
    e = c["e_rule"]
    # fusion_v1 stores one alpha (M1b + handcrafted); fusion_v2 stores a weights dict (+ M5).
    if "weights" in c:
        weights = {d: float(w) for d, w in c["weights"].items()}
    else:
        alpha = float(c["alpha_handcrafted"])
        weights = {"m1b_v3": 1 - alpha, "handcrafted_v5": alpha}
    refs = {d: np.asarray(v, dtype=float) for d, v in c["rank_ref_inner_oof_sorted"].items()}

    itw_companion = {"handcrafted_v5": str(S / "_itw_handcrafted_v5.csv"),
                     "m5_xlsr_ft": str(S / "_itw_m5.csv")}  # crop-fair M5 In-the-Wild, as in the v2 sweep
    cols = {d: load(d, itw_companion.get(d)) for d in weights}
    m1 = cols["m1b_v3"]
    m3 = load("spectra_aasist", str(REPO / "outputs" / "spectra" / "itw_stress" / "scores.csv"))
    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv")[["path", "label"]]
    itwm = pd.read_csv(REPO / "outputs" / "manifests" / "itw_stress.csv")[["path", "label"]]
    labels = pd.concat([lab, itwm]); labels["path"] = labels.path.map(norm)
    labels = labels.drop_duplicates("path").set_index("path").label
    gate = gate_flags()

    report: dict[str, dict] = {"constants": str(args.constants.resolve().relative_to(REPO)), "rule": c.get("final"),
                               "weights": weights, "e_rule": e, "splits": {}}  # fmt: skip
    lines = ["| Split | Configuration | minDCF (brief cost) | minDCF (sponsor-code cost) | Files E touched | Files G touched |",
             "|---|---|---|---|---|---|"]
    for split in SPLITS:
        idx = m1[m1.split == split].index.intersection(m3.index)
        for s in cols.values():
            idx = idx.intersection(s.index)
        ranks = {d: rank_of(cols[d].loc[idx, "logit"].to_numpy(), refs[d]) for d in weights}
        l3 = m3.loc[idx, "logit"].to_numpy()
        r3 = rank_of(l3, np.sort(m3[m3.split == "inner_oof"].logit.to_numpy()))
        base = np.zeros(len(idx))
        for d, w in weights.items():  # accumulated in the constants' order, as the runner does
            base = base + w * ranks[d]
        e_fires = (l3 < e["m3_logit_below"]) & (base > e["base_rank_above"])
        with_e = np.where(e_fires, base * e["multiply_by"], base)
        is_speech = gate.reindex(idx).fillna(1.0).to_numpy()
        g_fires = is_speech == 0
        configs = {
            "router_on": apply_default_answer(with_e, is_speech, keys=list(idx)),
            "no_gate": with_e,
            "no_suppression": apply_default_answer(base, is_speech, keys=list(idx)),
            "router_off": base,
            # equal-weight rank mean over every weighted column plus M3 as a full voter
            "fuse_everything": (sum(ranks.values()) + r3) / (len(ranks) + 1),
        }
        y = None
        if split != "test":
            y = (labels.reindex(idx) == "spoof").to_numpy(int)
        rows = {}
        for name, s in configs.items():
            if y is not None:
                rows[name] = {"brief": round(brief(y, s), 4), "averse": round(averse(y, s), 4)}
            else:
                thr = BLOCK_TOP + (1 - BLOCK_TOP) * 0.5
                rows[name] = {"share_above_half": round(float((s >= (thr if name in ("router_on", "no_suppression") else 0.5)).mean()), 4)}
            rows[name]["e_touched"] = int(e_fires.sum()) if name in ("router_on", "no_gate") else 0
            rows[name]["g_touched"] = int(g_fires.sum()) if name in ("router_on", "no_suppression") else 0
        ev = evidence_flags(idx) if split != "itw" else {"enf": -1, "splice": -1, "speaker_drift": -1}
        report["splits"][split] = {"n": len(idx), "n_spoof": int(y.sum()) if y is not None else None,
                                   "configs": rows, "evidence_only_flags_no_score_change": ev,
                                   "e_fires_by_label": ({"bonafide": int(e_fires[y == 0].sum()), "spoof": int(e_fires[y == 1].sum())}
                                                        if y is not None else None)}  # fmt: skip
        for name, r in rows.items():
            if y is not None:
                lines.append(f"| {split} (n={len(idx)}) | {name} | {r['brief']:.4f} | {r['averse']:.4f} | {r['e_touched']} | {r['g_touched']} |")
            else:
                lines.append(f"| {split} (n={len(idx)}) | {name} | share above 0.5: {r['share_above_half']:.1%} | | {r['e_touched']} | {r['g_touched']} |")
    md = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    args.out.with_suffix(".md").write_text(md + "\n")
    print(md)
    for split, r in report["splits"].items():
        print(f"{split}: evidence-only flags (no score change): {r['evidence_only_flags_no_score_change']}; E fires by label: {r['e_fires_by_label']}")
    print(f"wrote {args.out.with_suffix('.json')} and .md")


if __name__ == "__main__":
    main()
