"""Assemble one AntiDeepfake backbone's V2-53 exports into an Interface v1 score
file for the V2-01 benchmark.

  outputs/v2/backbone/<model>__inner.csv    (16,142 rows, split inner_oof, fold 0-4)
  outputs/v2/backbone/<model>__holdout.csv  ( 3,858 rows, split holdout)
  outputs/v2/backbone/<model>__test.csv     ( 1,671 rows, split test)
      -> outputs/v2/benchmark/scores/<model>.csv        (path,fold,split,score,logit)
  outputs/v2/backbone/<model>__itw.csv      ( 3,000 rows, split itw)
      -> outputs/v2/benchmark/scores/_itw_<model>.csv   (path,logit,split)

Every count is asserted, flagged rows (a decode failure scored as silence) are
refused, and the logit column is the "ours" view (band-matched, as v1). A
zero-shot model's inner rows are trivially out-of-fold; the fold column is
still carried so the benchmark's fold table and nested references work.

  uv run python scripts/v2_backbone_to_benchmark.py --model xls-r-2b-anti-deepfake
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hearsay import benchmark as B

LANE = REPO / "outputs/v2/backbone"
OUT = REPO / "outputs/v2/benchmark/scores"
EXPECT = {"inner": (B.N_INNER, "inner_oof"), "holdout": (B.N_HOLDOUT, "holdout"),
          "test": (B.N_TEST, "test"), "itw": (B.N_ITW, "itw")}


def load_part(model: str, pop: str, lane: Path = LANE) -> pd.DataFrame:
    p = lane / f"{model}__{pop}.csv"
    if not p.exists():
        raise B.BenchmarkError(f"{p} is missing: score it first (scripts/v2_backbone_score.py --population {pop})")
    df = pd.read_csv(p, dtype={"fold": str, "flag": str}, keep_default_na=False)
    n, split = EXPECT[pop]
    if len(df) != n:
        raise B.BenchmarkError(f"{p}: {len(df)} rows, expected {n}")
    if set(df["split"]) != {split}:
        raise B.BenchmarkError(f"{p}: split values {sorted(set(df['split']))}, expected {{{split!r}}}")
    if "flag" in df.columns and (df["flag"].astype(str).str.strip() != "").any():
        bad = int((df["flag"].astype(str).str.strip() != "").sum())
        raise B.BenchmarkError(f"{p}: {bad} flagged rows (decode failures scored as silence); repair before assembling")
    if df["path"].duplicated().any():
        raise B.BenchmarkError(f"{p}: duplicate paths")
    if pop == "inner" and set(df["fold"]) - {"0", "1", "2", "3", "4"}:
        raise B.BenchmarkError(f"{p}: inner folds {sorted(set(df['fold']))}, expected 0-4")
    return df[["path", "fold", "split", "score", "logit"]]


def assemble(model: str, *, lane: Path = LANE, out: Path = OUT) -> tuple[Path, Path]:
    main = pd.concat([load_part(model, p, lane) for p in ("inner", "holdout", "test")], ignore_index=True)
    itw = load_part(model, "itw", lane)[["path", "logit", "split"]]
    out.mkdir(parents=True, exist_ok=True)
    f_main, f_itw = out / f"{model}.csv", out / f"_itw_{model}.csv"
    main.to_csv(f_main, index=False)
    itw.to_csv(f_itw, index=False)
    return f_main, f_itw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    a = ap.parse_args(argv)
    f_main, f_itw = assemble(a.model)
    print(f"wrote {f_main} and {f_itw}")
    print(f"next: uv run python scripts/v2_benchmark.py score --file {f_main} --name {a.model.replace('-', '_')} "
          f"--itw {f_itw} --holdout --lane v2-53")
    return 0


if __name__ == "__main__":
    sys.exit(main())
