"""NSA training manifest: DiffSSD (spoof) + LJ Speech and LibriSpeech (bona fide).

Columns: path, label (spoof|bonafide), generator, speaker, utt, source.
- DiffSSD `generated_speech/<generator>/[speaker_<id>/]<file>`: single-speaker LJ-voice models
  (diffgan_tts, grad_tts, pro_diff, wavegrad2) have no speaker dir -> speaker "lj".
- LJ Speech: the full corpus (data/ljspeech/wavs) if present, else the NSA resampled subset
  (data/nsa/LJRealResampled). Speaker "lj", utt = LJ id; an id is kept once.
- LibriSpeech dev-clean/test-clean (data/librispeech): speaker = reader id.

Writes outputs/manifests/nsa_train_full.csv and, with --sample, a class-balanced subsample
nsa_train_sample.csv capped per generator (spoof) and per source (bona fide).

Usage: uv run python scripts/manifest_nsa.py [--per-generator 1000] [--bona-lj 6000]
       [--bona-libri 4000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
AUDIO = {".wav", ".mp3", ".flac"}


def diffssd_rows() -> list[dict]:
    root = DATA / "nsa" / "DiffSSD" / "generated_speech"
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in AUDIO:
            continue
        rel = p.relative_to(root).parts
        gen = rel[0]
        spk = rel[1] if len(rel) == 3 else "lj"
        rows.append({"path": str(p), "label": "spoof", "generator": gen, "speaker": spk,
                     "utt": p.stem, "source": "diffssd"})  # fmt: skip
    return rows


def lj_rows() -> list[dict]:
    full = DATA / "ljspeech" / "wavs"
    subset = DATA / "nsa" / "LJRealResampled" / "resampled"
    files = sorted(full.glob("*.wav")) if full.exists() else sorted(subset.glob("*.wav"))
    return [{"path": str(p), "label": "bonafide", "generator": "bonafide", "speaker": "lj",
             "utt": p.stem, "source": "ljspeech"} for p in files]  # fmt: skip


def libri_rows() -> list[dict]:
    root = DATA / "librispeech"
    return [{"path": str(p), "label": "bonafide", "generator": "bonafide",
             "speaker": f"libri_{p.parts[-3]}", "utt": p.stem, "source": "librispeech"}
            for p in sorted(root.rglob("*.flac"))]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-generator", type=int, default=1000)
    ap.add_argument("--bona-lj", type=int, default=6000)
    ap.add_argument("--bona-libri", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.DataFrame(diffssd_rows() + lj_rows() + libri_rows())
    assert not df.path.duplicated().any()
    out = REPO / "outputs" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "nsa_train_full.csv", index=False)
    print(f"full: {len(df)} clips")
    print(df.groupby(["label", "source", "generator"]).agg(n=("path", "size"),
          speakers=("speaker", "nunique")).to_string())  # fmt: skip

    spoof = df[df.label == "spoof"]
    parts = [g.sample(min(len(g), args.per_generator), random_state=args.seed)
             for _, g in spoof.groupby("generator")]  # fmt: skip
    for src, n in (("ljspeech", args.bona_lj), ("librispeech", args.bona_libri)):
        g = df[(df.label == "bonafide") & (df.source == src)]
        if len(g):
            parts.append(g.sample(min(len(g), n), random_state=args.seed))
    samp = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    samp.to_csv(out / "nsa_train_sample.csv", index=False)
    print(f"sample: {len(samp)} clips {samp.label.value_counts().to_dict()}, "
          f"{samp[samp.label == 'spoof'].generator.nunique()} generators, "
          f"{samp[samp.label == 'bonafide'].speaker.nunique()} bona fide speakers")


if __name__ == "__main__":
    main()
