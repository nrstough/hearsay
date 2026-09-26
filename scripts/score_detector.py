"""Run a rule-based (non-learned) contract detector over the training sample and the NSA test
set and export its scores in the fusion (M4) format.

Output: outputs/detector_scores/<name>.csv with path, fold, split, score, logit. Nothing is
fitted, so inner rows carry the detector's plain output (split=inner_oof, for uniformity with
the learned detectors), holdout rows split=holdout, test rows split=test. Rows whose result
is not "ok" keep NaN score/logit (fusion imputes them) and are counted. A second file,
outputs/detector_scores/<name>_results.csv, keeps status, evidence and features per row for
the explanation report and the docs.

Usage: uv run python scripts/score_detector.py --detector container \
    --folds splits/nsa_folds.csv --test-manifest outputs/manifests/nsa_test.csv [--workers 6]
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.detectors.base import REGISTRY, ClipContext, safe_run
from hearsay.metrics import eer, min_cost

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="module hearsay.detectors.<detector>")
    ap.add_argument("--name", help="registry name (default: --detector)")
    ap.add_argument("--folds", type=Path, default=REPO / "splits" / "nsa_folds.csv")
    ap.add_argument("--test-manifest", type=Path,
                    default=REPO / "outputs" / "manifests" / "nsa_test.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N training rows (smoke test)")
    args = ap.parse_args()

    importlib.import_module(f"hearsay.detectors.{args.detector}")
    det = REGISTRY.get(args.name or args.detector)
    f = pd.read_csv(args.folds)
    if args.limit:
        f = f.iloc[: args.limit]
    t = pd.read_csv(args.test_manifest)
    rows = pd.concat([
        f.assign(split=np.where(f.fold == "holdout", "holdout", "inner_oof")),
        t.assign(fold="test", split="test", label=np.nan),
    ], ignore_index=True)  # fmt: skip
    paths = [p if Path(p).is_absolute() else str(REPO / p) for p in rows.path]

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda p: safe_run(det, ClipContext(Path(p))), paths))
    ok = np.array([r.status == "ok" for r in res])
    score = np.array([r.score if r.status == "ok" else np.nan for r in res])
    s = np.clip(score, 1e-4, 1 - 1e-4)
    logit = np.log(s / (1 - s))
    export = pd.DataFrame({"path": rows.path, "fold": rows.fold, "split": rows.split,
                           "score": score, "logit": logit})  # fmt: skip
    assert export.path.is_unique
    out = REPO / "outputs" / "detector_scores"
    out.mkdir(parents=True, exist_ok=True)
    export.to_csv(out / f"{det.name}.csv", index=False)
    results = export.assign(
        status=[r.status for r in res], evidence=[r.evidence for r in res],
        error=[r.error or "" for r in res],
        features=[json.dumps(r.features, sort_keys=True) for r in res],
    )  # fmt: skip
    results.to_csv(out / f"{det.name}_results.csv", index=False)

    summary = {"detector": det.name, "rows": len(rows), "ok": int(ok.sum()),
               "not_ok": int((~ok).sum()), "seconds": round(time.time() - t0)}  # fmt: skip
    for split in ("inner_oof", "holdout"):
        m = (rows.split == split).to_numpy() & ok
        y = (rows.label[m] == "spoof").to_numpy(int)
        if m.sum() and 0 < y.sum() < m.sum():
            summary[split] = {"n": int(m.sum()), "score_std": round(float(score[m].std()), 4),
                              "min_dcf": round(min_cost(y, score[m]), 4),
                              "eer": round(eer(y, score[m]), 4)}  # fmt: skip
    m = (rows.split == "test").to_numpy() & ok
    summary["test"] = {"n": int(m.sum()), "score_mean": round(float(np.nanmean(score[m])), 4),
                       "score_std": round(float(np.nanstd(score[m])), 4),
                       "distinct_scores": int(pd.Series(score[m]).nunique())}  # fmt: skip
    (out / f"{det.name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out / (det.name + '.csv')}")


if __name__ == "__main__":
    main()
