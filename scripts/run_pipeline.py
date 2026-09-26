"""End-to-end runner (the Docker entrypoint): a directory of audio -> the shipped detectors ->
the persisted fusion rule (or the M1 probe alone) -> the non-speech gate -> one AnalyzeResponse
JSON per file, a resumable results.jsonl cache (headed by the model identities it was scored
with; a different probe, bundle, constants or code starts fresh), the NSA TSV, a sidecar and
run_meta.json, all under --out. Never writes into submissions/ and never appends to the submission log: those
belong to the main chat. CLI contract: docker/entrypoint.sh (run spec
docs/specs/2026-09-26_k-docker-image.md, D4/D4a/D10); `hearsay.pipeline` does the scoring.

Listing (D4a): with --template (NSA's HearsayScoreKey4TeamX.tsv, tab-separated, `filename`
column) rows keep the template's order; each name resolves as a relative path under --data,
else by basename against a recursive index of --data (nested layouts work; the same basename in
two directories is ambiguous and fails before scoring). A template name absent from --data, or
a file that does not decode, still gets a row (the default answer, `undetermined`, flag
`missing_file` / `decode_error`), so the TSV is always complete. Duplicate template names fail
before scoring; an empty listing fails with no TSV. Without a template: sorted audio files.

Scoring: --detectors m1b alone means M1 only, P = sigmoid(LLR + logit(0.3)) exactly as
scripts/make_probe_csv.py; otherwise every fused detector runs and --rule is read from the
constants file, never refit: e_on_a (default; models/fusion_v1/constants.json from
scripts/fuse_sweep.py: 0.8 rank(M1b) + 0.2 rank(handcrafted), M3 as false-alarm suppression only)
or zmean | stack_nonlj (models/fusion_v0/constants.json from scripts/fuse.py, via --fusion). A rule
the file does not define is refused. The policy then maps determinate scores to [0.001, 1] and
pins gated files below 0.001 (hearsay.detectors.speech_gate.apply_default_answer, applied exactly
once); --flip emits the pre-flipped variant (1 - p mapped the same way). A rerun with the same
--out resumes from results.jsonl; another --rule, --policy or --flip re-fuses the cached logits
without running a model.

Usage:
  uv run python scripts/run_pipeline.py --data data/nsa/HackGTHearsayTesting --out outputs/run1 \
      --template data/nsa/HearsayScoreKey4TeamX.tsv --team HEARSAY --limit 50 \
      [--rule e_on_a] [--flip] [--compare-tsv submissions/<logged>.tsv]
  docker run --network none -v <test_dir>:/data:ro -v <out_dir>:/out hearsay    # entrypoint.sh
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from hearsay import SR
from hearsay.detectors.base import ClipContext
from hearsay.detectors.speech_gate import BLOCK_TOP
from hearsay.metrics import PI_SYNTH
from hearsay.pipeline import (
    DEFAULT_CONSTANTS_PATH,
    DEFAULT_RULE,
    DETECTOR_ORDER,
    HC_DIR,
    M1_EMBED_FP16,
    M1_MODES,
    PROBE_DIR,
    REPO,
    RULES,
    SCORER_FLAGS,
    FusionConstants,
    Models,
    analyze_clip,
    final_score,
    fuse_items,
    fusion_block,
    git_sha,
    refuse,
    verdict_for,
)
from hearsay.submission import AUDIO_EXT, write_submission

ALL_SCORERS = "m1b,spectra,handcrafted"


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--in", dest="data", type=Path, required=True, help="directory of audio files")
    ap.add_argument("--out", type=Path, required=True, help="output directory (created)")
    ap.add_argument("--template", type=Path, default=_env_path("HEARSAY_TEMPLATE"),
                    help="TSV whose `filename` column fixes the row order ($HEARSAY_TEMPLATE)")  # fmt: skip
    ap.add_argument("--team", default=os.environ.get("HEARSAY_TEAM", "HEARSAY"),
                    help="team name for <team>_predictions.tsv ($HEARSAY_TEAM)")  # fmt: skip
    ap.add_argument("--detectors", default=os.environ.get("HEARSAY_DETECTORS", ALL_SCORERS),
                    help=f"comma list of {sorted(SCORER_FLAGS)}; m1b alone = M1 only, no fusion ($HEARSAY_DETECTORS)")  # fmt: skip
    ap.add_argument("--fusion", "--constants", dest="fusion", type=Path, default=_env_path("HEARSAY_FUSION"),
                    help="fusion constants file ($HEARSAY_FUSION; default <app-root>/models/fusion_v1/"
                         "constants.json when more than m1b is requested; models/fusion_v0/constants.json "
                         "holds zmean and stack_nonlj)")  # fmt: skip
    ap.add_argument("--rule", default=os.environ.get("HEARSAY_RULE", DEFAULT_RULE), choices=RULES,
                    help=f"fusion rule; must be one the constants file defines (default {DEFAULT_RULE})")  # fmt: skip
    ap.add_argument("--flip", action="store_true",
                    help="emit the pre-flipped variant: 1 - p through the same determinate map, block "
                         "still at the minimum (only if NSA's scoring turns out inverted)")  # fmt: skip
    ap.add_argument("--policy", default=os.environ.get("HEARSAY_POLICY", "speech_gate"),
                    choices=["speech_gate", "none"],
                    help="none: leave the fused probability ungated (parity against pre-gate TSVs)")  # fmt: skip
    ap.add_argument("--limit", type=int, help="score only the first N rows (run_meta.json says partial)")
    ap.add_argument("--device", default="cpu", choices=["cpu"], help="CPU only")
    ap.add_argument("--m1-mode", default="segment", choices=M1_MODES,
                    help="segment (prepare_segment -> embed_segment, the export's path) | windows (embed_clip) | auto (= segment)")  # fmt: skip
    ap.add_argument("--no-truncate", action="store_true",
                    help="keep all 24 XLS-R layers (default: drop the layers above the probe's; "
                         "hidden_states[7] is bit-identical, about 3x faster)")  # fmt: skip
    ap.add_argument("--fresh", action="store_true", help="ignore results.jsonl (moved aside) and rescore")
    ap.add_argument("--require-offline", action="store_true",
                    help="refuse to run unless HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 (the image sets them)")  # fmt: skip
    ap.add_argument("--app-root", type=Path, default=REPO,
                    help="base directory holding models/ (probe, hc_selected, fusion_v0) inside the image")  # fmt: skip
    ap.add_argument("--pi-synth", type=float, default=PI_SYNTH, help="prior shift for the M1-only posterior")
    ap.add_argument("--threads", type=int, help="torch threads (default: $OMP_NUM_THREADS, else min(cores, 6))")
    ap.add_argument("--probe", type=Path, help="probe dir (default <app-root>/models/m1_shipped or the pinned 0521 probe)")
    ap.add_argument("--hc", type=Path, help="handcrafted bundle dir (default <app-root>/models/hc_selected)")
    ap.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True,
                    help="score silence and a chord twice first; finite and reproducible")  # fmt: skip
    ap.add_argument("--no-tsv", action="store_true", help="skip the TSV (JSON and cache only)")
    ap.add_argument("--compare-tsv", type=Path, help="a logged TSV to compare probabilities against")
    return ap.parse_args(argv)


def resolve_scorers(detectors: str, fusion: Path | None, app_root: Path) -> tuple[tuple[str, ...], Path | None]:
    """--detectors and --fusion -> (fused columns to compute, constants path or None for M1 only)."""
    flags = [f.strip() for f in detectors.split(",") if f.strip()]
    bad = [f for f in flags if f not in SCORER_FLAGS]
    if bad or not flags:
        raise ValueError(f"--detectors must be a comma list of {sorted(SCORER_FLAGS)}, got {detectors!r}")
    scorers = tuple(d for d in DETECTOR_ORDER if d in {SCORER_FLAGS[f] for f in flags})
    if fusion is None and scorers == ("m1b_v3",):
        return scorers, None  # M1 only
    if fusion is None:
        fusion = app_root / DEFAULT_CONSTANTS_PATH.relative_to(REPO)
    if scorers != DETECTOR_ORDER:
        print(f"note: fusion needs every fused detector ({', '.join(DETECTOR_ORDER)}), "
              f"not only --detectors {detectors!r}; running all of them", flush=True)  # fmt: skip
    return DETECTOR_ORDER, fusion


def default_probe(app_root: Path) -> Path:
    shipped = app_root / "models" / "m1_shipped"  # docker/build.sh's name for the shipped probe
    if (shipped / "probe.joblib").exists():
        return shipped
    pinned = app_root / "models" / PROBE_DIR.name
    return pinned if (pinned / "probe.joblib").exists() or app_root != REPO else PROBE_DIR


def read_template_ids(template: Path) -> list[str]:
    """`filename` column of NSA's tab-separated template; header required; names unique. Anything
    that is not tab-separated fails loudly (a comma reader would swallow the whole header)."""
    text = template.read_text().splitlines()
    if not text:
        raise ValueError(f"{template}: empty template")
    if "\t" not in text[0]:
        raise ValueError(f"{template}: not tab-separated (header: {text[0]!r}); NSA's template is a TSV")
    rows = list(csv.DictReader(text, delimiter="\t"))
    if not rows or "filename" not in rows[0]:
        raise ValueError(f"{template}: no `filename` column (header: {text[0]!r})")
    ids = [r["filename"] for r in rows]
    if len(set(ids)) != len(ids):
        seen, dup = set(), None
        for i in ids:
            if i in seen:
                dup = i
                break
            seen.add(i)
        raise ValueError(f"{template}: duplicate filenames (first: {dup})")
    return ids


def index_audio(data: Path) -> dict[str, list[Path]]:
    """basename -> paths, over every audio file under `data` (recursive, sorted)."""
    idx: dict[str, list[Path]] = {}
    for p in sorted(data.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            idx.setdefault(p.name, []).append(p)
    return idx


def list_inputs(data: Path, template: Path | None) -> list[tuple[str, Path | None]]:
    """(id, path or None when missing) in template order, else sorted relative paths.
    Raises ValueError on a duplicate template name or an ambiguous basename."""
    if template is None:
        files = sorted(p for p in data.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)
        return [(str(p.relative_to(data)), p) for p in files]
    idx = index_audio(data)
    out: list[tuple[str, Path | None]] = []
    for name in read_template_ids(template):
        direct = data / name
        if direct.is_file():
            out.append((name, direct))
            continue
        hits = idx.get(Path(name).name, [])
        if len(hits) > 1:
            raise ValueError(f"{name}: ambiguous, {len(hits)} files share that basename under {data}: "
                             + ", ".join(str(h.relative_to(data)) for h in hits[:3]))  # fmt: skip
        out.append((name, hits[0] if hits else None))
    return out


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cache_identity(probe_dir: Path, hc_dir: Path, fusion: Path | None, scorers, m1_mode: str,
                   truncate: bool) -> dict:  # fmt: skip
    """What a cached row's logits depend on. A resume requires an exact match (rule and policy
    are not in it: those are re-fused from the cached logits)."""
    return {"_header": "hearsay.run_pipeline results.jsonl", "git_sha": git_sha(),
            "probe": probe_dir.resolve().name, "hc": hc_dir.resolve().name if hc_dir.exists() else hc_dir.name,
            "constants_sha": _sha256(fusion), "scorers": list(scorers), "m1_mode": m1_mode,
            "m1_fp16": M1_EMBED_FP16, "truncate": truncate}  # fmt: skip


def load_cache(path: Path) -> tuple[dict | None, dict[str, dict]]:
    """(header, filename -> latest cached response); a torn last line (crash mid-write) is
    dropped. header is None for a missing file or a cache written before headers existed."""
    cache: dict[str, dict] = {}
    header = None
    if not path.exists():
        return header, cache
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, dict):
            continue
        if "_header" in doc and header is None:
            header = doc
        elif "filename" in doc:
            cache[doc["filename"]] = doc
    return header, cache


def open_cache(path: Path, identity: dict, fresh: bool) -> dict[str, dict]:
    """Rows to reuse. A cache whose header differs from `identity` (other probe, bundle,
    constants, code or settings), a header-less cache, or --fresh: moved aside, nothing reused.
    Writes the header when the file is new."""
    header, cache = load_cache(path)
    same = header is not None and all(header.get(k) == v for k, v in identity.items() if k != "created")
    if path.exists() and (fresh or not same):
        why = "--fresh" if fresh else ("no header" if header is None else "identity changed: " + ", ".join(
            f"{k} {header.get(k)!r} -> {v!r}" for k, v in identity.items() if header.get(k) != v))
        aside = path.with_name(f"{path.name}.stale-{_stamp()}")
        k = 2
        while aside.exists():  # two moves within one second must not overwrite each other
            aside = path.with_name(f"{path.name}.stale-{_stamp()}-{k}")
            k += 1
        path.rename(aside)
        print(f"cache {path} not reused ({why}); moved to {aside.name}", flush=True)
        cache = {}
    if not path.exists():
        with path.open("w") as f:
            f.write(json.dumps({**identity, "created": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
    return cache


PREFLIGHT_ABS_TOL = 1e-6


def preflight_consistent(pa: float, pb: float, abs_tol: float = PREFLIGHT_ABS_TOL) -> bool:
    """Two preflight passes agree: both finite and within `abs_tol`. Not `==`: multithreaded
    x86 BLAS/oneDNN is not run-to-run bit-reproducible (the amd64 image saw 8e-10 between two
    passes on silence); the arm64 Mac happens to be exact. hearsay.submission.preflight's own
    `a.scores == b.scores` is not used here for the same reason."""
    return (math.isfinite(pa) and math.isfinite(pb)
            and math.isclose(pa, pb, rel_tol=0.0, abs_tol=abs_tol))  # fmt: skip


def preflight(models: Models, consts: FusionConstants | None, rule: str, tmp: Path,
              apply_gate: bool = True, **kw) -> dict:  # fmt: skip
    """plan.md: silence and a chord through the whole pipeline, twice; finite and reproducible
    (within PREFLIGHT_ABS_TOL). Both are non-speech, so with the gate on both must land in the
    pinned block below BLOCK_TOP (exact checks)."""
    tmp.mkdir(parents=True, exist_ok=True)
    t = np.arange(4 * SR) / SR
    chord = sum(0.1 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    sf.write(tmp / "silence.wav", np.zeros(4 * SR, dtype=np.float32), SR)
    sf.write(tmp / "music.wav", chord.astype(np.float32), SR)
    out = {}
    for name in ("silence.wav", "music.wav"):
        a = analyze_clip(ClipContext(tmp / name), models, consts, rule, apply_gate=apply_gate, **kw)
        b = analyze_clip(ClipContext(tmp / name), models, consts, rule, apply_gate=apply_gate, **kw)
        pa, pb = a["probability_synthetic"], b["probability_synthetic"]
        if not preflight_consistent(pa, pb):
            raise RuntimeError(f"preflight {name}: not finite or not reproducible: {pa} vs {pb}")
        if a["is_speech"]:
            raise RuntimeError(f"preflight {name}: the speech gate let a non-speech clip through")
        if apply_gate and not pa < BLOCK_TOP:
            raise RuntimeError(f"preflight {name}: gated file not in the pinned block below {BLOCK_TOP}: {pa}")
        out[name] = {"p": pa, "p_fused": a["fusion"]["p_fused"], "verdict": a["verdict"],
                     "inputs": a["fusion"]["inputs"]}  # fmt: skip
        (tmp / f"{name}.json").write_text(json.dumps(a))
    print(f"preflight: silence={out['silence.wav']['p']:.4f} music={out['music.wav']['p']:.4f} "
          f"(before the gate: {out['silence.wav']['p_fused']:.3f}, {out['music.wav']['p_fused']:.3f})",
          flush=True)  # fmt: skip
    return out


def compare_tsv(ids: list[str], scores: list[float], ref: Path) -> dict:
    """Spearman and max |diff| of our probabilities against a logged TSV, on shared filenames."""
    from scipy.stats import spearmanr

    ref_rows = {r["filename"]: float(r["cm-score"])
                for r in csv.DictReader(ref.read_text().splitlines(), delimiter="\t")}  # fmt: skip
    pairs = [(s, ref_rows[i]) for i, s in zip(ids, scores, strict=True) if i in ref_rows]
    if len(pairs) < 3:
        return {"ref": str(ref), "n": len(pairs), "note": "too few shared files"}
    ours, theirs = (np.array(v, dtype=np.float64) for v in zip(*pairs, strict=True))
    d = np.abs(ours - theirs)
    return {"ref": str(ref), "n": len(pairs),
            "spearman": round(float(spearmanr(ours, theirs).statistic), 6),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "n_diff_gt_0.01": int((d > 0.01).sum()), "n_diff_gt_0.05": int((d > 0.05).sum())}  # fmt: skip


def missing_doc(fid: str, consts: FusionConstants | None, rule: str, apply_gate: bool,
                pi_synth: float, version: dict, flip: bool = False) -> dict:  # fmt: skip
    """The row for a template name with no file under --data: no detector ran, the file sits
    below FAILURE_TOP like a decode failure (with --policy none, the all-imputed fused probability
    through the determinate map)."""
    fo = fuse_items([], consts, rule, pi_synth)
    p = final_score(fo.p, False, apply_gate, key=fid, order_by=fo.p, failed=True, flip=flip)
    return {"filename": fid, "duration_s": None, "probability_synthetic": p,
            "verdict": verdict_for(p, False, False), "is_speech": False,
            "default_answer_applied": apply_gate, "fusion": fusion_block(fo), "detectors": [],
            "routing_log": [f"missing: {fid} is not under the data directory; no detector ran; "
                            + "the default answer applies"],
            "flag": "missing_file", "seconds": 0.0, "version": version}  # fmt: skip


def write_sidecar(path: Path, ids: list[str], docs: dict[str, dict]) -> None:
    cols = ["filename", "probability_synthetic", "flag", "is_speech", *DETECTOR_ORDER, "fused", "p_fused", "seconds"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in ids:
            d, fu = docs[i], docs[i]["fusion"]
            w.writerow([i, d["probability_synthetic"], d.get("flag", ""), d["is_speech"],
                        *[fu["inputs"].get(n, "") for n in DETECTOR_ORDER], fu["fused"], fu["p_fused"],
                        d.get("seconds", "")])  # fmt: skip


def _stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def main(argv=None) -> int:
    args = parse_args(argv)
    t_run = time.time()
    if args.require_offline:
        missing = [k for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE") if os.environ.get(k) != "1"]
        if missing:
            sys.exit(f"--require-offline: set {' and '.join(missing)}=1 (the image does); refusing to run")
    if not args.data.is_dir():
        sys.exit(f"--data {args.data} is not a directory")
    try:
        scorers, fusion = resolve_scorers(args.detectors, args.fusion, args.app_root)
        items = list_inputs(args.data, args.template)
    except ValueError as e:
        sys.exit(f"error: {e}")
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.exit(f"no audio files under {args.data}" + (f" (template {args.template})" if args.template else ""))
    n_missing = sum(p is None for _, p in items)
    if n_missing:
        print(f"warning: {n_missing} template name(s) not found under {args.data}; they get the "
              f"default answer with flag missing_file", flush=True)  # fmt: skip
    probe_dir = args.probe or default_probe(args.app_root)
    hc_dir = args.hc or (args.app_root / "models" / "hc_selected" if args.app_root != REPO else HC_DIR)
    threads = args.threads or (int(os.environ["OMP_NUM_THREADS"]) if os.environ.get("OMP_NUM_THREADS") else None)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    cache_path = out / "results.jsonl"
    cache = open_cache(cache_path, cache_identity(probe_dir, hc_dir, fusion, scorers, args.m1_mode,
                                                 not args.no_truncate), args.fresh)  # fmt: skip
    todo = [(i, p) for i, p in items if i not in cache]
    consts = None
    if fusion is not None:
        if not fusion.exists():
            sys.exit(f"fusion constants {fusion} not found (run scripts/fuse.py, or --detectors m1b)")
        consts = FusionConstants.load(fusion)
        if args.rule not in consts.rules():
            sys.exit(f"rule {args.rule!r} is not defined by {fusion} (it has {consts.rules()}); "
                     f"pass --rule from that list or another --fusion file")  # fmt: skip
    rule = args.rule if consts is not None else "m1b_only"
    apply_gate = args.policy == "speech_gate"
    kw = {"scorers": scorers, "pi_synth": args.pi_synth, "flip": args.flip}
    print(f"{len(items)} files ({len(items) - len(todo)} cached in {cache_path}); scorers {', '.join(scorers)}; "
          f"rule {rule}; policy {args.policy}; polarity {'flipped' if args.flip else 'our_direction'}; order from "
          f"{'template ' + str(args.template) if args.template else 'sorted filenames'}", flush=True)  # fmt: skip

    version = {"git_sha": git_sha(), "fusion": fusion.name if fusion else "none", "rule": rule,
               "policy": args.policy, "polarity": "flipped" if args.flip else "our_direction",
               "scorers": list(scorers), "m1_mode": args.m1_mode,
               "fusion_final": getattr(consts, "final", None)}  # fmt: skip
    build_info = args.app_root / "BUILD_INFO"
    if build_info.exists():
        version["build_info"] = build_info.read_text().strip()
    meta: dict = {"started": datetime.now().astimezone().isoformat(timespec="seconds"),
                  "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  "n_files": len(items), "n_cached": len(items) - len(todo), "n_missing_file": n_missing,
                  "partial": bool(args.limit)}  # fmt: skip
    models = None
    if any(p is not None for _, p in todo) or args.preflight:
        t0 = time.time()
        models = Models(device=args.device, probe_dir=probe_dir, hc_dir=hc_dir, threads=threads,
                        m1_mode=args.m1_mode, load_spectra="spectra_aasist" in scorers,
                        truncate=not args.no_truncate)  # fmt: skip
        version["models"] = models.version()
        meta["model_load_seconds"] = round(time.time() - t0, 2)
        meta["threads"] = models.threads
        print(f"models loaded in {meta['model_load_seconds']}s ({models.load_seconds}), "
              f"{models.threads} threads; probe {probe_dir}, hc {hc_dir}", flush=True)  # fmt: skip
        if args.preflight:
            meta["preflight"] = preflight(models, consts, rule, out / "preflight", apply_gate, **kw)

    timings_path = out / "timings.csv"
    new_timings = not timings_path.exists()
    docs: dict[str, dict] = {}
    t_score, secs = time.time(), []
    with cache_path.open("a") as cache_f, timings_path.open("a", newline="") as tf:
        tw = csv.writer(tf)
        if new_timings:
            tw.writerow(["filename", "seconds", "engineered_s", "m1b_v3_s", "spectra_aasist_s",
                         "probability_synthetic", "is_speech", "flag"])  # fmt: skip
        for k, (fid, path) in enumerate(items, start=1):
            if fid in cache:
                doc = cache[fid]
                v = doc.get("version", {})
                if (doc.get("fusion", {}).get("rule") != rule or v.get("policy", "speech_gate") != args.policy
                        or v.get("polarity", "our_direction") != version["polarity"]):
                    try:
                        doc = refuse(doc, consts, rule, apply_gate=apply_gate, pi_synth=args.pi_synth,
                                     flip=args.flip)  # fmt: skip
                    except (KeyError, ValueError):
                        doc = None  # the cache lacks a logit this rule needs: rescore below
                    else:  # keep the per-file JSON in step with the TSV being written
                        (out / "results" / f"{Path(fid).name}.json").write_text(json.dumps(doc, indent=1))
                if doc is not None:
                    docs[fid] = doc
                    continue
            t0 = time.time()
            if path is None:
                doc = missing_doc(fid, consts, rule, apply_gate, args.pi_synth, version, flip=args.flip)
            else:
                doc = analyze_clip(ClipContext(path), models, consts, rule, version=version,
                                   apply_gate=apply_gate, **kw)  # fmt: skip
                doc["filename"] = fid  # the template's id, which may include a subdirectory
            dt = time.time() - t0
            secs.append(dt)
            cache_f.write(json.dumps(doc) + "\n")
            cache_f.flush()
            (out / "results" / f"{Path(fid).name}.json").write_text(json.dumps(doc, indent=1))
            by = {d["name"]: d["seconds"] for d in doc["detectors"]}
            eng = sum(v for n, v in by.items() if n not in ("m1b_v3", "spectra_aasist"))
            tw.writerow([fid, round(dt, 3), round(eng, 3), by.get("m1b_v3", ""), by.get("spectra_aasist", ""),
                         doc["probability_synthetic"], doc["is_speech"], doc.get("flag", "")])  # fmt: skip
            tf.flush()
            docs[fid] = doc
            if k % 25 == 0 or k == len(items):
                el = time.time() - t_score
                print(f"  {k}/{len(items)}  {el:.0f}s  {np.mean(secs):.2f}s/file  "
                      f"eta {(len(todo) - len(secs)) * np.mean(secs) / 60:.1f} min", flush=True)  # fmt: skip

    ids = [i for i, _ in items]
    scores = [float(docs[i]["probability_synthetic"]) for i in ids]
    meta["scored_seconds"] = round(time.time() - t_score, 1)
    if secs:
        meta["per_file_seconds"] = {"n": len(secs), "mean": round(float(np.mean(secs)), 3),
                                    "median": round(float(np.median(secs)), 3),
                                    "p95": round(float(np.percentile(secs, 95)), 3),
                                    "max": round(float(np.max(secs)), 3)}  # fmt: skip
    flags = [docs[i].get("flag", "") for i in ids]
    meta["n_gated"] = int(sum(not docs[i]["is_speech"] for i in ids))
    meta["flags"] = {f: flags.count(f) for f in sorted(set(flags)) if f}
    meta["share_gt_0.5"] = round(float(np.mean(np.array(scores) > 0.5)), 4)
    meta["version"] = version

    stem = f"{args.team}_predictions" + ("_FLIPPED" if args.flip else "")
    if not args.no_tsv:
        tsv = out / f"{stem}.tsv"
        if tsv.exists():  # a new file per run, never an overwrite
            stem = f"{stem}-{_stamp()}"
            tsv = out / f"{stem}.tsv"
        write_submission(ids, scores, tsv)
        meta["tsv"] = str(tsv)
        print(f"wrote {tsv} ({len(ids)} rows)", flush=True)
    write_sidecar(out / f"{stem}.sidecar.csv", ids, docs)
    if args.compare_tsv:
        meta["compare"] = compare_tsv(ids, scores, args.compare_tsv)
        print("compare:", json.dumps(meta["compare"]), flush=True)
    meta["wall_seconds"] = round(time.time() - t_run, 1)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {len(ids)} files, {meta['n_gated']} gated, flags {meta['flags']}, "
          f"share>0.5 {meta['share_gt_0.5']}, wall {meta['wall_seconds']}s "
          f"({meta.get('per_file_seconds', {}).get('mean', 'cached')} s/file)", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
