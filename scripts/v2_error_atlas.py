"""V2-02 error atlas of v1 on the sealed NSA test key.

Reads the labels ONLY through the persisted Phase 0 sealed readout
(outputs/v2/benchmark/sealed/SHIP_tsv.json): the error list and the worst-N
lists it carries. Never opens the key file. Everything else is unlabelled
per-file material the v1 runner already produced for every test clip:

  outputs/runner/v2_full/HEARSAY_predictions.sidecar.csv   fused inputs, p, is_speech
  outputs/runner/v2_full/results/<file>.json               detector features and routing log
  splits/nsa_test_durations.csv                            durations
  outputs/detector_scores/wavlm_l.csv                      the WavLM Large probe (not shipped)

Writes outputs/v2/error_atlas.csv (one row per listed clip, features joined)
and prints the markdown tables for docs/reports/2026-10-20_v2-02-error-atlas.md.

  uv run python scripts/v2_error_atlas.py [--sealed outputs/v2/benchmark/sealed/SHIP_tsv.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hearsay import benchmark as B

RUN = REPO / "outputs/runner/v2_full"
COLUMNS = ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist", "wavlm_l")
FEATURES = {  # detector -> features copied from the per-file JSON
    "compression": ("bw_hz", "band_7_8k_db", "hf_slope_db_per_khz", "deep_hole_frac"),
    "speech_gate": ("voiced_frac", "rms_dbfs", "flatness"),
    "speaker_drift": ("cos_min", "drift"),
    "splice": ("n_seams",),
    "enf": ("enf_present",),
}


def per_file(name: str) -> dict:
    d = json.loads((RUN / "results" / f"{name}.json").read_text())
    row = {"duration_s": d["duration_s"], "default_answer": d["default_answer_applied"],
           "e_applied": d["fusion"]["detail"].get("e_applied")}
    for det in d["detectors"]:
        if det["name"] in FEATURES:
            for f in FEATURES[det["name"]]:
                row[f"{det['name']}.{f}"] = det.get("features", {}).get(f)
    m = re.search(r"voiced (\d+)% of frames", " ".join(d["routing_log"]))
    row["voiced_pct_log"] = int(m.group(1)) if m else None
    return row


def test_frame() -> pd.DataFrame:
    """All 1,671 test clips, unlabelled: fused inputs, p, features, WavLM logit, and each column's
    percentile within the test set (share of the set scoring strictly below)."""
    side = pd.read_csv(RUN / "HEARSAY_predictions.sidecar.csv").set_index("filename")
    dur = pd.read_csv(REPO / "splits/nsa_test_durations.csv").set_index("filename")["duration_s"]
    wl = pd.read_csv(REPO / "outputs/detector_scores/wavlm_l.csv")
    wl = wl[wl["split"] == "test"].assign(filename=lambda d: d["path"].map(lambda p: Path(p).name)).set_index("filename")
    df = side.join(dur.rename("duration_manifest_s"), how="left").join(wl["logit"].rename("wavlm_l"), how="left")
    if len(df) != B.N_TEST or df[list(COLUMNS)].isna().any().any():
        raise SystemExit("sidecar / WavLM export do not cover the 1,671 test clips")
    feats = pd.DataFrame.from_dict({n: per_file(n) for n in df.index}, orient="index")
    df = df.join(feats)
    for c in COLUMNS:
        df[f"pct.{c}"] = df[c].rank(method="average", pct=True)
    df["pct.shipped"] = df["probability_synthetic"].rank(method="average", pct=True)
    return df


def listed_rows(readout: dict) -> pd.DataFrame:
    rows = []
    for kind, lst in (("false_alarm", readout["errors_at_decision"]["false_alarms"]),
                      ("miss", readout["errors_at_decision"]["misses"]),
                      ("worst_real", readout["worst"]["real_highest"]),
                      ("worst_spoof", readout["worst"]["spoof_lowest"])):
        for r in lst:
            rows.append({"kind": kind, **r})
    df = pd.DataFrame(rows)
    # a clip can be both an error and in a worst list; keep the error row and note the other membership
    df["in_worst"] = df.groupby("filename")["kind"].transform(lambda k: ",".join(sorted(set(k))))
    return df[~df["kind"].str.startswith("worst") | ~df["filename"].duplicated(keep="last")].drop_duplicates("filename")


def md_table(df: pd.DataFrame, cols: list, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                cells.append("NA" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sealed", default=str(REPO / "outputs/v2/benchmark/sealed/SHIP_tsv.json"))
    ap.add_argument("--out", default=str(REPO / "outputs/v2/error_atlas.csv"))
    a = ap.parse_args(argv)
    card = json.loads(Path(a.sealed).read_text())
    if card.get("role") != "sealed" or not card["sealed_ledger"].startswith(("read", "persisted")):
        raise SystemExit("not a ledgered sealed readout")
    r = card["readout"]
    n_spoof, n_real = r["n_spoof"], r["n_real"]
    spoof_share = n_spoof / (n_spoof + n_real)
    top_cut = 1 - spoof_share  # a column "ranks it synthetic" if its percentile is above the prior's cut

    allf = test_frame()
    listed = listed_rows(r).set_index("filename").join(allf, how="left")
    listed.to_csv(a.out)
    err = listed[listed["kind"].isin(("miss", "false_alarm"))]
    miss = listed[listed["kind"] == "miss"]

    print(f"# atlas inputs: sealed read {card['sealed_ledger']} (phase {card['phase']}); "
          f"{len(err)} errors at 0.5, {len(listed)} listed clips, {len(allf)} test clips\n")
    print("## Readings (shipped TSV on the key)\n")
    p = r["pooled"]
    print(f"brief {p['brief']:.4f} | package-0.5 {p['package05']:.4f} | package-0.3 {p['package03']:.4f} | "
          f"EER {p['eer']:.4f}; bootstrap p5/p50/p95 package-0.3 = "
          + "/".join(f"{r['bootstrap']['stratified']['package03'][k]:.4f}" for k in ("p5", "p50", "p95")))
    for k, c in r["counts"].items():
        print(f"- {k}: min-cost threshold {c['threshold']:.4f}: {c['n_fa']} FA / {c['n_miss']} misses")
    c = r["counts_at_decision"]
    print(f"- at 0.5: {c['n_fa']} FA / {c['n_miss']} misses (P_FA {c['p_fa']:.4%}, P_miss {c['p_miss']:.2%})\n")

    print("## The errors at 0.5\n")
    cols = ["kind", "score", "duration_s", "is_speech", "e_applied", "m1b_v3", "handcrafted_v5", "m5_xlsr_ft",
            "spectra_aasist", "wavlm_l", "compression.bw_hz", "speech_gate.voiced_frac", "speaker_drift.cos_min"]
    print(md_table(err.reset_index(), ["filename"] + cols, {"score": lambda v: f"{v:.4f}"}))

    print("\n## Which column ranks each miss in the top synthetic share "
          f"(percentile > {top_cut:.3f}, the key's real share)\n")
    rank_cols = [f"pct.{c}" for c in COLUMNS]
    caught = (miss[rank_cols] > top_cut)
    tbl = miss[["score"] + rank_cols].copy()
    tbl["caught_by"] = caught.apply(lambda row: ",".join(c.split(".")[1] for c, v in row.items() if v) or "none", axis=1)
    print(md_table(tbl.reset_index(), ["filename", "score"] + rank_cols + ["caught_by"], {"score": lambda v: f"{v:.4f}"}))
    print("\ncounts: " + ", ".join(f"{c.split('.')[1]} {int(caught[c].sum())}/{len(miss)}" for c in rank_cols)
          + f"; any column {int(caught.any(axis=1).sum())}/{len(miss)}; none {int((~caught.any(axis=1)).sum())}/{len(miss)}")

    print("\n## Misses vs the whole test set (medians)\n")
    feat = ["duration_s", "compression.bw_hz", "compression.band_7_8k_db", "speech_gate.voiced_frac",
            "speech_gate.rms_dbfs", "speaker_drift.cos_min", "m1b_v3", "m5_xlsr_ft", "spectra_aasist", "wavlm_l"]
    med = pd.DataFrame({f"misses (n={len(miss)})": miss[feat].median(), f"all test (n={len(allf)})": allf[feat].median(),
                        "miss p25": miss[feat].quantile(0.25), "miss p75": miss[feat].quantile(0.75)})
    med.index.name = "feature"
    print(md_table(med.reset_index(), list(med.reset_index().columns)))
    print(f"\nshort clips (< 3.5 s): misses {int((miss['duration_s'] < 3.5).sum())}/{len(miss)} "
          f"({(miss['duration_s'] < 3.5).mean():.0%}) vs test {(allf['duration_s'] < 3.5).mean():.0%}; "
          f"E suppression fired on {int(miss['e_applied'].fillna(False).astype(bool).sum())} misses; "
          f"speech gate false on {int((~miss['is_speech'].astype(bool)).sum())} misses; "
          f"bandwidth at the 7.25 kHz band match or below on {int((miss['compression.bw_hz'] <= 7250).sum())} misses "
          f"(test {(allf['compression.bw_hz'] <= 7250).mean():.0%})")

    print("\n## The worst-scoring real clips (highest shipped score)\n")
    wr = listed[listed["kind"].isin(("false_alarm", "worst_real"))].sort_values("score", ascending=False).head(12)
    print(md_table(wr.reset_index(), ["filename", "score", "duration_s", "m1b_v3", "handcrafted_v5", "m5_xlsr_ft",
                                      "spectra_aasist", "wavlm_l", "compression.bw_hz", "speech_gate.voiced_frac",
                                      "speaker_drift.cos_min"], {"score": lambda v: f"{v:.4f}"}))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
