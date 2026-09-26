"""Pull an English MLAAD subset sampled by generator, not volume (plan.md "Data").

MLAAD (mueller91/MLAAD, gated, non-commercial notice) is spoof-only: TTS renderings of M-AILABS
book utterances. For each English model folder we read its meta.csv, keep clips >= --min-dur
seconds (NSA clips are > 2 s), and sample --per-model clips. Output:
  data/mlaad/fake/en/<model>/*.wav           (gitignored)
  outputs/manifests/mlaad_en.csv             path,label,generator,model_name,speaker,source_file,...

`generator` is MLAAD's `architecture` column (the grouping key for generator-held-out splits);
`source_file` is the M-AILABS original, so bona fide M-AILABS clips can be kept utterance-disjoint.

Usage:
  uv run python scripts/pull_mlaad.py [--per-model 30] [--min-dur 2.0]
  # generators named on the NSA kickoff slide, deeper sample:
  uv run python scripts/pull_mlaad.py --models "ElevenLabs|Gemini|Qwen3" --per-model 300 \
      --name mlaad_en_named
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "mueller91/MLAAD"
DEST = REPO / "data" / "mlaad"


def read_meta(model_dir: str) -> pd.DataFrame:
    p = hf_hub_download(REPO_ID, f"{model_dir}/meta.csv", repo_type="dataset", local_dir=DEST)
    df = pd.read_csv(p, sep="|", dtype=str)
    df["model_dir"] = model_dir
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--per-model", type=int, default=30)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", help="regex on model folder name (default: all)")
    ap.add_argument("--name", help="manifest name (default: mlaad_<lang>)")
    args = ap.parse_args()

    api = HfApi()
    dirs = [x.path for x in api.list_repo_tree(REPO_ID, f"fake/{args.lang}", repo_type="dataset")]
    if args.models:
        dirs = [d for d in dirs if re.search(args.models, d.rsplit("/", 1)[-1], re.IGNORECASE)]
    print(f"{len(dirs)} {args.lang} model folders; reading meta.csv files")
    with ThreadPoolExecutor(16) as ex:
        meta = pd.concat(list(ex.map(read_meta, dirs)), ignore_index=True)

    meta["duration"] = pd.to_numeric(meta["duration"], errors="coerce")
    meta = meta[meta.duration >= args.min_dur]
    samp = pd.concat(
        g.sample(min(len(g), args.per_model), random_state=args.seed)
        for _, g in meta.groupby("model_dir")
    )
    rel = [p.removeprefix("./") for p in samp.path]
    print(f"downloading {len(rel)} clips from {samp.model_dir.nunique()} models "
          f"({samp.architecture.nunique()} architectures)")
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=DEST, allow_patterns=rel,
                      max_workers=16)  # fmt: skip

    parts = samp.original_file.str.split("/")
    man = pd.DataFrame({
        "path": [str(DEST / r) for r in rel],
        "label": "spoof",
        "generator": samp.architecture.to_numpy(),
        "model_name": samp.model_name.to_numpy(),
        "speaker": parts.str[3].to_numpy(),  # en_UK/by_book/<gender>/<speaker>/<book>/wavs/x.wav
        "source_file": samp.original_file.to_numpy(),
        "duration": samp.duration.to_numpy(),
        "transcript": samp.transcript.to_numpy(),
        "dataset": "mlaad",
    })  # fmt: skip
    missing = sum(not Path(p).exists() for p in man.path)
    out = REPO / "outputs" / "manifests" / f"{args.name or 'mlaad_' + args.lang}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(out, index=False)
    print(f"wrote {out}: {len(man)} clips, {man.generator.nunique()} architectures, "
          f"{man.speaker.nunique()} source speakers, missing files {missing}")


if __name__ == "__main__":
    main()
