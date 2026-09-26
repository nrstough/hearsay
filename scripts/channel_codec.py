"""Action B of the channel-robustness rung: is the NSA test set's 7.2 kHz wall a codec?
(run spec docs/specs/2026-09-26_channel-robustness.md, D4-D5)

100 inner-fold real clips (50 LJ, 50 LibriSpeech; seed 0) go through a grid: raw, the
pipeline's Kaiser low-pass alone, and MP3/AAC round-trips (hearsay.compression.launder) at
several bitrates and encode rates, each alone and followed by the Kaiser low-pass. The
statistics describe the unfiltered result (what the file is, not what the pipeline does to
it) and are measured the same way on all 1,671 test files. Distance = mean over statistics of
|median(variant) - median(test)| / IQR(test). Pre-declared match rule (D5): a codec variant
matches if its distance is >= 20% below the Kaiser-alone distance AND it is closer to the
test medians than Kaiser on both hole statistics.

Usage: uv run python scripts/channel_codec.py [--workers 6] [--n 50]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "channel"
SR = 16000
N_FFT = 1024
EDGES = (6500, 6750, 7000, 7250, 7500, 7750)  # 250 Hz bands, as outputs/inventory/nsa_test_highband.csv
HOLE_STATS = ("deep_hole_frac", "local_hole_frac")
STATS = ("lvl_6500", "lvl_6750", "lvl_7000", "lvl_7250", "lvl_7500", "lvl_7750", "drop_7500_vs_6500",
         "rolloff_slope_db_per_khz", "hb_flatness_6_7k", *HOLE_STATS, "floor_p2_db", "clip_floor_db")  # fmt: skip
MATCH_IMPROVEMENT = 0.20


def variant_grid() -> list[str]:
    """Variant names: raw, kaiser, <codec>-<kbps>k@<sr>, and the same with a +kaiser suffix."""
    codecs = [f"mp3-{k}k@{sr}" for k in (24, 32, 48, 64, 96) for sr in (16000, 44100)]
    codecs += [f"aac-{k}k@{sr}" for k in (32, 48, 64) for sr in (16000, 44100)]
    return ["raw", "kaiser", *codecs, *[f"{c}+kaiser" for c in codecs]]


def apply_variant(x: np.ndarray, name: str) -> np.ndarray:
    from hearsay.compression import launder, parse_laundering
    from hearsay.handcrafted import band_limit

    if name == "raw":
        return x
    if name == "kaiser":
        return band_limit(x)
    codec, _, post = name.partition("+")
    y = launder(x, *parse_laundering(codec))
    return band_limit(y) if post == "kaiser" else y


def highband_stats(x: np.ndarray) -> dict[str, float]:
    """Band levels at 6.5-8 kHz relative to the 1-3 kHz mean of the long-term spectrum, the
    7.5-vs-6.5 kHz drop, the roll-off slope over 6.5-8 kHz, 6-7 kHz spectral flatness, and the
    compression detector's hole and floor statistics (no band match, no crop). Silence is
    trimmed first so codec padding and corpus lead silence do not enter."""
    import librosa

    from hearsay.audio import trim_silence
    from hearsay.compression import features

    x = trim_silence(np.asarray(x, dtype=np.float32))
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=256)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    lt = 10 * np.log10(S.mean(axis=1) + 1e-12)
    ref = float(lt[(fr >= 1000) & (fr <= 3000)].mean())
    f: dict[str, float] = {}
    for lo in EDGES:
        m = (fr >= lo) & (fr < lo + 250)
        f[f"lvl_{lo}"] = float(lt[m].mean() - ref)
    f["drop_7500_vs_6500"] = f["lvl_7500"] - f["lvl_6500"]
    m = (fr >= 6500) & (fr < 8000)
    f["rolloff_slope_db_per_khz"] = float(np.polyfit(fr[m] / 1000, lt[m], 1)[0])
    b = S[(fr >= 6000) & (fr < 7000)] + 1e-12
    f["hb_flatness_6_7k"] = float(np.mean(np.exp(np.mean(np.log(b), axis=0)) / np.mean(b, axis=0)))
    c = features(x, None, None, "segment", False)
    for k in ("deep_hole_frac", "local_hole_frac", "floor_p2_db", "clip_floor_db"):
        f[k] = c[k]
    return {k: (v if np.isfinite(v) else 0.0) for k, v in f.items()}


def match_distance(var: pd.DataFrame, test: pd.DataFrame, stats=STATS) -> tuple[float, list[str]]:
    """Mean |median difference| in units of the test IQR; statistics with zero test IQR are
    excluded and returned."""
    ds, excluded = [], []
    for s in stats:
        iqr = float(test[s].quantile(0.75) - test[s].quantile(0.25))
        if not iqr > 1e-9:
            excluded.append(s)
            continue
        ds.append(abs(float(var[s].median()) - float(test[s].median())) / iqr)
    return (float(np.mean(ds)) if ds else float("nan")), excluded


def codec_match(table: pd.DataFrame) -> dict:
    """table: one row per variant with `distance` and |median - test median| per hole statistic
    (`gap_<stat>`). Applies the pre-declared rule against the `kaiser` row."""
    k = table.set_index("variant").loc["kaiser"]
    cands = table[~table.variant.isin(["raw", "kaiser"])].copy()
    cands["better_distance"] = cands.distance <= (1 - MATCH_IMPROVEMENT) * k.distance
    cands["better_holes"] = np.logical_and.reduce([cands[f"gap_{s}"] < k[f"gap_{s}"] for s in HOLE_STATS])
    ok = cands[cands.better_distance & cands.better_holes].sort_values("distance")
    best = cands.sort_values("distance").iloc[0] if len(cands) else None
    return {
        "kaiser_distance": round(float(k.distance), 4),
        "match": bool(len(ok)), "matched_variant": ok.iloc[0].variant if len(ok) else None,
        "matched_distance": round(float(ok.iloc[0].distance), 4) if len(ok) else None,
        "closest_variant": None if best is None else best.variant,
        "closest_distance": None if best is None else round(float(best.distance), 4),
        "n_matching": len(ok), "matching": list(ok.variant[:10]),
    }  # fmt: skip


def _clip(args: tuple[str, tuple[str, ...]]) -> list[dict]:
    from hearsay.audio import load_audio

    path, variants = args
    x = load_audio(path)
    out = []
    for v in variants:
        try:
            out.append({"path": path, "variant": v, **highband_stats(apply_variant(x, v))})
        except Exception as e:  # noqa: BLE001 - one failed variant is a missing row, counted
            out.append({"path": path, "variant": v, "error": repr(e)[:200]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=50, help="clips per real source")
    ap.add_argument("--from-rows", action="store_true", help="rebuild the tables from codec_grid_rows.csv")
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.from_rows:
        d = pd.read_csv(OUT / "codec_grid_rows.csv")
        return summarize(d[d.error.isna()] if "error" in d else d, variant_grid(), t0)

    f = pd.read_csv(REPO / "splits" / "nsa_folds.csv")
    inner = f[(f.fold != "holdout") & (f.label == "bonafide")]
    src = pd.concat([inner[inner.source == s].sample(a.n, random_state=0) for s in ("ljspeech", "librispeech")])
    grid = variant_grid()
    test = pd.read_csv(REPO / "outputs" / "manifests" / "nsa_test.csv")
    tpaths = [str((REPO / p).resolve()) for p in test.path]

    jobs = [(p, tuple(grid)) for p in src.path] + [(p, ("raw",)) for p in tpaths]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_clip, jobs, chunksize=4)):
            rows.extend(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} clips, {time.time() - t0:.0f} s", flush=True)
    d = pd.DataFrame(rows)
    d["set"] = np.where(d.path.isin(set(tpaths)), "test", "ref")
    d.to_csv(OUT / "codec_grid_rows.csv", index=False)
    if "error" in d and d.error.notna().any():
        print(f"{int(d.error.notna().sum())} failed variant rows; e.g. {d[d.error.notna()].iloc[0].to_dict()}")
        d = d[d.error.isna()]
    summarize(d, grid, t0)


def grid_table(d: pd.DataFrame, grid: list[str], stats=STATS) -> pd.DataFrame:
    tst = d[d.set == "test"]
    table = []
    for v in grid:
        var = d[(d.set == "ref") & (d.variant == v)]
        if var.empty:
            continue
        dist, excl = match_distance(var, tst, stats)
        row = {"variant": v, "n": len(var), "distance": dist, "excluded": ";".join(excl)}
        for s in STATS:
            row[f"med_{s}"] = float(var[s].median())
        for s in HOLE_STATS:
            row[f"gap_{s}"] = abs(float(var[s].median()) - float(tst[s].median()))
        table.append(row)
    return pd.DataFrame(table).sort_values("distance")


def summarize(d: pd.DataFrame, grid: list[str], t0: float) -> None:
    tst = d[d.set == "test"]
    table = grid_table(d, grid)
    table.to_csv(OUT / "codec_grid.csv", index=False)
    res = codec_match(table)
    # Sensitivity: without clip_floor_db, a whole-spectrogram percentile that reads the 7-8 kHz
    # stopband (the column action A had to drop).
    res["without_clip_floor_db"] = codec_match(grid_table(d, grid, tuple(x for x in STATS if x != "clip_floor_db")))
    res["test_medians"] = {s: round(float(tst[s].median()), 3) for s in STATS}
    res["top10"] = table.head(10)[["variant", "distance", *[f"med_{s}" for s in ("drop_7500_vs_6500",
                                   "rolloff_slope_db_per_khz", *HOLE_STATS, "clip_floor_db")]]].round(3).to_dict("records")  # fmt: skip
    res["raw_and_kaiser"] = table[table.variant.isin(["raw", "kaiser"])][["variant", "distance"]].round(4).to_dict("records")
    res["seconds"] = round(time.time() - t0)
    (OUT / "codec_match.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
