"""Write a submission TSV from precomputed test embeddings (no GPU needed).

Same math as scripts/make_probe_csv.py (P = sigmoid(LLR + logit(pi_synth))), but the LLRs
come from a bulk embedding set produced by scripts/extract_embeddings.py with the probe's own
mode (segment mode: prepare_segment, i.e. band match + trim + 8 s cap). Parity with the
per-clip scorer was checked on nsa_test_v2 (max |diff| 0.0017, MPS float noise). Row order
comes from the NSA template. The silence/music preflight runs through the per-clip path on CPU.

Usage:
  uv run python scripts/tsv_from_embeddings.py --probe models/m1_... --embeddings nsa_test_v3 \
      --val-mindcf 0.159 --val-eer 0.0285 --notes "..."
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.embed import embed_clip, embed_segment, load_backbone, prepare_segment
from hearsay.metrics import sigmoid
from hearsay.probe import Probe, load_embeddings
from hearsay.submission import REPO, append_log, preflight, write_submission

TEMPLATE = REPO / "data" / "nsa" / "HearsayScoreKey4TeamX.tsv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--embeddings", required=True, help="test embedding set name")
    ap.add_argument("--pi-synth", type=float, default=0.3)
    ap.add_argument("--val-mindcf", default="")
    ap.add_argument("--val-eer", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    probe = Probe.load(args.probe)
    X, m = load_embeddings(probe.backbone, args.embeddings)
    shift = math.log(args.pi_synth / (1 - args.pi_synth))
    llr = probe.llr(X)
    by_name = pd.Series(sigmoid(llr + shift), index=[os.path.basename(p) for p in m.path])
    raw = pd.Series(llr, index=by_name.index)
    ids = pd.read_csv(TEMPLATE, sep="\t").filename.tolist()
    missing = [i for i in ids if i not in by_name.index]
    assert not missing, f"{len(missing)} template files have no embedding, e.g. {missing[:3]}"

    if not args.skip_preflight:
        model = load_backbone(probe.backbone, "cpu")

        def score(x: np.ndarray) -> float:
            e = (embed_segment(model, prepare_segment(x)) if probe.segment
                 else embed_clip(model, x, probe.win_s, probe.max_windows))  # fmt: skip
            return float(sigmoid(probe.llr(e)[0] + shift))

        tmp = REPO / "outputs" / "preflight"
        tmp.mkdir(parents=True, exist_ok=True)
        preflight(score, float(sigmoid(shift)), tmp)

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    name = f"{stamp}_M1_{args.probe.name}"
    out = REPO / "submissions" / f"{name}.tsv"
    scores = [float(by_name[i]) for i in ids]
    write_submission(ids, scores, out)
    side = REPO / "outputs" / "submissions" / f"{name}.sidecar.csv"
    side.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "p": scores, "llr": [float(raw[i]) for i in ids]}).to_csv(
        side, index=False
    )
    s = np.array(scores)
    notes = (f"minDCF={args.val_mindcf} EER={args.val_eer}; probe {args.probe.name}; "
             f"from embeddings {args.embeddings}; pi_synth={args.pi_synth}; "
             f"share>0.5={np.mean(s > 0.5):.3f} >0.9={np.mean(s > 0.9):.3f}")  # fmt: skip
    append_log(
        "M1", out, validation_score=args.val_mindcf, notes=f"{notes}. {args.notes}".strip(". ")
    )
    print(f"wrote {out}: {len(ids)} rows; share>0.5 {np.mean(s > 0.5):.3f}, "
          f">0.9 {np.mean(s > 0.9):.3f}; sidecar {side}")


if __name__ == "__main__":
    main()
