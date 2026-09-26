"""ASVspoof 2019 LA -> outputs/manifests/asv19_{train,dev,eval}.csv.

Columns: path, label (spoof|bonafide), speaker, generator (attack id A01-A19, "bonafide" for
bona fide), partition. Train and dev share attacks A01-A06; eval uses A07-A19 (A16 = A04 and
A19 = A06 by design), so eval is the generator-held-out split for public-data work.

Usage: uv run python scripts/manifest_asvspoof19.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LA = REPO / "data" / "asvspoof2019" / "LA"
PROTO = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def main() -> None:
    out_dir = REPO / "outputs" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    for part, fname in PROTO.items():
        cols = ["speaker", "utt", "_", "generator", "label"]
        df = pd.read_csv(LA / "ASVspoof2019_LA_cm_protocols" / fname, sep=" ", names=cols)
        df["path"] = [str(LA / f"ASVspoof2019_LA_{part}" / "flac" / f"{u}.flac") for u in df.utt]
        df["generator"] = df.generator.replace("-", "bonafide")
        df["partition"] = part
        missing = sum(not Path(p).exists() for p in df.path)
        df = df[["path", "label", "speaker", "generator", "partition"]]
        df.to_csv(out_dir / f"asv19_{part}.csv", index=False)
        bal = df.label.value_counts().to_dict()
        gens = sorted(set(df.generator) - {"bonafide"})
        print(f"{part}: {len(df)} clips {bal}, generators {gens}, missing files {missing}")


if __name__ == "__main__":
    main()
