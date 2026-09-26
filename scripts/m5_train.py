"""M5 trainer (runs on the rented box; `--cpu-smoke` runs the same code on a tiny model).

One invocation trains one model: `--fold k` (validation = core fold k, out-of-fold export) or
`--fold full` (all inner folds; exports holdout + test). Selection is wall-clock fixed
(`--steps`), never a checkpoint pick on the validation fold. Every exported score uses the
deployment transform (whole trimmed clip up to 14 s, fp32); test-length crops and the
augmented holdout slice are diagnostics only.

Refuses: no CUDA (unless --cpu-smoke), bundle tree-sha or XLS-R config sha mismatch, any
trivial-feature shortcut AUC > --shortcut-max on THIS model's training rows.

Usage (box): python scripts/m5_train.py --bundle /root/m5/bundle --fold 4 --arm nsa_extra \
    --steps 3000 --out /root/m5/runs/fold4 --r2-prefix r2:pa-source/hearsay/runs/fold4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from hearsay import SR
from hearsay.augment import Augmenter, holdout_slice, trim_jitter
from hearsay.m5_bundle import read_clip, tree_sha
from hearsay.m5_data import (
    DEPLOY_MAX_S,
    LengthMixer,
    band_match,
    bundle_transform,
    collate,
    crop,
    epoch_rng,
    holdout_rows,
    shortcut_aucs,
    training_rows,
    validation_rows,
)
from hearsay.m5_model import (
    M5Config,
    bce_with_smoothing,
    build_model,
    param_groups,
    save_m5,
    score_batch,
)
from hearsay.metrics import report, sigmoid

REPO = Path(__file__).resolve().parents[1]
R2_PREFIX = "r2:pa-source/hearsay/"
ARMS = {"nsa": ("core", "extra_nsa"), "nsa_extra": ("core", "extra_nsa", "extra_asv19", "extra_mlaad")}


# --- data -----------------------------------------------------------------------------------


class M5Dataset(Dataset):
    """Keys are (row_index, crop_s, epoch): one crop length per batch, re-seeded per epoch."""

    def __init__(self, rows: pd.DataFrame, root: Path, aug: Augmenter | None, seed: int,
                 variants: dict[str, str] | None):  # fmt: skip
        self.rows = rows.reset_index(drop=True)
        self.root = root
        self.aug = aug
        self.seed = seed
        self.variants = variants or {}
        self.files = [str(root / ("core" if s == "core" else "extra") / f)
                      for s, f in zip(self.rows.train_scope, self.rows.file)]
        self.labels = (self.rows.label == "spoof").to_numpy(np.int64)
        self.ids = self.rows.id.tolist()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        i, crop_s, epoch = key
        rng = epoch_rng(self.seed, epoch, i)
        ops = ""
        if self.aug is not None:
            plan = self.aug.plan(rng, has_codec=self.ids[i] in self.variants)
        else:
            plan = []
        if plan and any(op == "codec" for op, _ in plan):
            x = read_clip(self.variants[self.ids[i]])
        else:
            x = read_clip(self.files[i])
        if plan:
            x = trim_jitter(x, rng)
        x = crop(x, crop_s, rng)
        if plan:
            x = self.aug.apply(x, plan, rng)
            ops = "+".join(op for op, _ in plan)
        x = band_match(x)  # deterministic, every clip, after augmentation (as at test time)
        return x, int(self.labels[i]), i, ops


class MixBatchSampler:
    """Label-balanced batches with per-source quotas (spec D1 mix); infinite, seeded."""

    def __init__(self, rows: pd.DataFrame, batch_size: int, mix: dict, mixer: LengthMixer,
                 seed: int, steps_per_epoch: int):  # fmt: skip
        self.rng = np.random.default_rng(seed + 7)
        self.bs = batch_size
        self.mixer = mixer
        self.spe = steps_per_epoch
        self.pools: dict[tuple[str, str], np.ndarray] = {}
        for (label, src), d in rows.reset_index(drop=True).groupby(["label", "source"]):
            self.pools[(label, src)] = d.index.to_numpy()
        self.weights = {}
        for label, key in (("bonafide", "real"), ("spoof", "spoof")):
            w = {s: mix[key].get(s, 0.0) for (l_, s) in self.pools if l_ == label}
            tot = sum(w.values())
            if tot <= 0:  # sources not named in the mix: uniform
                w = {s: 1.0 for s in w}
                tot = sum(w.values())
            self.weights[label] = {s: v / tot for s, v in w.items()}

    def draw(self, label: str, n: int) -> list[int]:
        srcs = list(self.weights[label])
        p = np.array([self.weights[label][s] for s in srcs])
        counts = self.rng.multinomial(n, p)
        out = []
        for s, c in zip(srcs, counts):
            if c:
                out.extend(self.rng.choice(self.pools[(label, s)], size=c, replace=True).tolist())
        return out

    def batches(self, start_step: int, steps: int):
        for step in range(start_step, steps):
            crop_s = self.mixer.draw()
            epoch = step // self.spe
            half = self.bs // 2
            idx = self.draw("bonafide", half) + self.draw("spoof", self.bs - half)
            yield [(i, crop_s, epoch) for i in idx]


def collate_batch(items):
    xs, mask = collate([it[0] for it in items])
    return (torch.from_numpy(xs), torch.from_numpy(mask),
            torch.tensor([it[1] for it in items]), [it[2] for it in items], [it[3] for it in items])


# --- helpers --------------------------------------------------------------------------------


def rclone_push(src: Path, dst: str) -> None:
    assert dst.startswith(R2_PREFIX), dst
    subprocess.run(["rclone", "copy", str(src), dst, "--transfers", "8"], check=False)


def rclone_pull(src: str, dst: Path) -> None:
    assert src.startswith(R2_PREFIX), src
    subprocess.run(["rclone", "copy", src, str(dst), "--transfers", "8"], check=False)


def lr_lambda(step: int, total: int, warmup: float, schedule: str) -> float:
    w = max(1, int(warmup * total))
    if step < w:
        return (step + 1) / w
    if schedule == "cosine":
        t = (step - w) / max(1, total - w)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return max(0.0, 1 - (step - w) / max(1, total - w))


@torch.no_grad()
def score_rows(net, files, transform, device, batch=16, max_s=14.0):
    """fp32 logits for a list of FLAC paths through `transform` (callable on the raw clip)."""
    net.eval()
    out = np.full(len(files), np.nan)
    order = np.argsort([-os.path.getsize(f) for f in files])  # long first: fewer pad waste
    for b in range(0, len(order), batch):
        ii = order[b : b + batch]
        clips = [transform(read_clip(files[i])) for i in ii]
        xs, mask = collate(clips)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            enabled=False):
            out[ii] = score_batch(net, xs, mask, device).numpy()
    return out


def export(paths, folds, split, logits) -> pd.DataFrame:
    paths = list(paths)
    folds = [folds] * len(paths) if isinstance(folds, str) else list(folds)
    return pd.DataFrame({"path": paths, "fold": folds, "split": split,
                         "score": sigmoid(logits), "logit": logits})  # fmt: skip


# --- main -----------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--fold", required=True, help="0..4 or full")
    ap.add_argument("--arm", choices=list(ARMS), default="nsa_extra")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=800, help="validation rows per eval (crops)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--r2-prefix", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deadline", type=float, default=None, help="unix time to stop training")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--skip-tree-check", action="store_true")
    ap.add_argument("--shortcut-max", type=float, default=0.85)
    ap.add_argument("--config", type=str, default="{}", help="JSON overrides for M5Config")
    args = ap.parse_args()

    cfg = M5Config(**json.loads(args.config))
    if args.cpu_smoke:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        sys.exit("FATAL: M5 trains on CUDA only (never the local GPU); use --cpu-smoke for tests")
    torch.manual_seed(cfg.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    log = open(out / "train_log.jsonl", "a")  # noqa: SIM115 (closed at exit)

    def jlog(**kv):
        kv["t"] = round(time.time(), 1)
        log.write(json.dumps(kv) + "\n")
        log.flush()
        print(json.dumps(kv), flush=True)

    # --- bundle identity ---
    meta = json.loads((args.bundle / "bundle_meta.json").read_text())
    if not args.skip_tree_check:
        got = tree_sha(args.bundle, exclude=("TREE_SHA", "bundle_meta.json", "codecs"))
        if got != meta["tree_sha"]:
            sys.exit(f"FATAL: bundle tree sha {got[:12]} != {meta['tree_sha'][:12]}")
    weights_dir = args.weights_dir or (REPO / "weights" / "wav2vec2-xls-r-300m")
    if not args.cpu_smoke:
        import hashlib

        got = hashlib.sha256((weights_dir / "config.json").read_bytes()).hexdigest()
        want = (args.bundle / "config_sha.txt").read_text().strip()
        if got != want:
            sys.exit(f"FATAL: XLS-R config sha {got[:12]} != bundled {want[:12]}")

    # --- rows ---
    m = pd.read_csv(args.bundle / "manifest.csv")
    m["fold"] = m["fold"].astype(str)
    fold = None if args.fold == "full" else str(args.fold)
    tr = training_rows(m, fold)
    tr = tr[tr.train_scope.isin(ARMS[args.arm])]
    cap = cfg.mix.get("mlaad_cap_per_model")
    if cap and (tr.train_scope == "extra_mlaad").any():
        rng = np.random.default_rng(cfg.seed)
        ml = tr[tr.train_scope == "extra_mlaad"]
        keep = ml.groupby("model_name", group_keys=False).apply(
            lambda d: d.sample(min(len(d), cap), random_state=int(rng.integers(1 << 30))))
        tr = pd.concat([tr[tr.train_scope != "extra_mlaad"], keep])
    tr = tr.reset_index(drop=True)
    assert not tr.fold.eq("holdout").any()

    # --- shortcut gate on THIS model's training rows only ---
    tr = tr.assign(dur14=np.minimum(tr.duration, 14.0))
    gate = shortcut_aucs(tr, features=("duration", "dur14", "peak", "lead_s"))
    worst = max(v for d in gate.values() for v in d.values() if v is not None)
    jlog(event="shortcut_gate", worst=worst, gate=gate, n_train=len(tr),
         by_scope=tr.groupby(["train_scope", "label"]).size().to_dict().__repr__())
    if worst > args.shortcut_max:
        sys.exit(f"FATAL: trivial-feature shortcut AUC {worst:.3f} > {args.shortcut_max}")

    variants = {}
    cm = args.bundle / "codecs" / "codec_manifest.csv"
    if cm.exists():
        c = pd.read_csv(cm)
        variants = {i: str(args.bundle / "codecs" / f) for i, f in zip(c.id, c.file)}
    aug = Augmenter(p_aug=cfg.p_aug, p_rawboost=cfg.p_rawboost)
    ds = M5Dataset(tr, args.bundle, aug, cfg.seed, variants)
    mixer = LengthMixer(cfg.seed, p_test=cfg.p_test_len)
    sampler = MixBatchSampler(tr, cfg.batch_size, cfg.mix, mixer, cfg.seed, args.steps_per_epoch)

    # validation slice: fold k's core rows (test-length crops, fixed seed); the full model has
    # no held-out fold, so it watches an in-sample slice for direction only
    if fold is not None:
        va = validation_rows(m, fold).reset_index(drop=True)
        va_note = f"fold {fold} (out of fold)"
    else:
        va = tr[tr.train_scope == "core"].sample(min(args.eval_n, len(tr)), random_state=0)
        va = va.reset_index(drop=True)
        va_note = "in-sample slice (direction only)"
    va = va.iloc[: args.eval_n] if len(va) > args.eval_n else va
    va_files = [str(args.bundle / "core" / f) for f in va.file]
    va_y = (va.label == "spoof").to_numpy(int)
    va_len = LengthMixer(1234, p_test=1.0)
    va_crops = [va_len.draw() for _ in range(len(va))]

    def val_transform_factory(j):
        return lambda x: band_match(crop(x, va_crops[j], np.random.default_rng([99, j])))

    # --- model ---
    net = build_model(cfg, weights_dir, tiny=args.cpu_smoke).to(device)
    opt = torch.optim.AdamW(param_groups(net, cfg))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.steps, cfg.warmup, cfg.schedule))
    step = 0
    ck = out / "ckpt" / "last.pt"
    if args.resume:
        if args.r2_prefix and not ck.exists():
            rclone_pull(args.r2_prefix.rstrip("/") + "/ckpt", out / "ckpt")
        if ck.exists():
            state = torch.load(ck, map_location=device, weights_only=False)
            net.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            step = int(state["step"])
            sampler.rng = np.random.default_rng(state["sampler_seed"])
            jlog(event="resumed", step=step)

    def checkpoint():
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "sampler_seed": int(sampler.rng.integers(1 << 30))}, ck)  # fmt: skip
        if args.r2_prefix:
            rclone_push(out / "ckpt", args.r2_prefix.rstrip("/") + "/ckpt")

    def evaluate(tag: str) -> dict:
        logits = np.full(len(va), np.nan)
        net.eval()
        for b in range(0, len(va), 16):
            ii = list(range(b, min(b + 16, len(va))))
            clips = [val_transform_factory(j)(read_clip(va_files[j])) for j in ii]
            xs, mask = collate(clips)
            logits[ii] = score_batch(net, xs, mask, device).numpy()
        net.train()
        r = report(va_y, logits)
        from sklearn.metrics import roc_auc_score

        r["auc"] = round(float(roc_auc_score(va_y, logits)), 4)
        w = net.layer_weights()
        jlog(event="eval", tag=tag, step=step, val=va_note, **r,
             layer_weights=None if w is None else [round(float(v), 4) for v in w])
        return r

    jlog(event="start", fold=args.fold, arm=args.arm, steps=args.steps, device=device,
         config=asdict(cfg), config_hash=cfg.hash(), bundle_tree=meta["tree_sha"],
         n_train=len(tr), n_val=len(va), val=va_note, trainable=sum(
             p.numel() for p in net.parameters() if p.requires_grad))

    # --- train ---
    loader = DataLoader(ds, batch_sampler=sampler.batches(step, args.steps),
                        num_workers=args.workers, collate_fn=collate_batch,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    aug_tally: Counter = Counter()
    tot_tally: Counter = Counter()
    net.train()
    t0, t_last, clips_seen = time.time(), time.time(), 0
    use_amp = device.startswith("cuda")
    for xs, mask, y, idx, ops in loader:
        if args.deadline and time.time() > args.deadline:
            jlog(event="deadline_stop", step=step)
            break
        xs, mask, y = xs.to(device, non_blocking=True), mask.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = net(xs, mask)
        loss = bce_with_smoothing(logits, y, cfg.label_smoothing)
        if not torch.isfinite(loss):
            jlog(event="nan_loss", step=step)
            sys.exit("FATAL: non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                       cfg.grad_clip)
        opt.step()
        sched.step()
        step += 1
        clips_seen += len(idx)
        for i, o in zip(idx, ops):
            key = (ds.rows.source.iat[i], ds.rows.label.iat[i])
            tot_tally[key] += 1
            if o:
                aug_tally[key] += 1
        if step % 50 == 0:
            now = time.time()
            jlog(event="train", step=step, loss=round(float(loss), 4),
                 clips_per_s=round(len(idx) * 50 / (now - t_last), 1),
                 crop_s=round(xs.shape[1] / SR, 2), lr=sched.get_last_lr()[-1])
            t_last = now
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate("periodic")
            if step >= 500 and r["auc"] <= 0.5 and fold is not None:
                sys.exit("FATAL: validation AUC <= 0.5 after 500 steps (direction tripwire)")
            checkpoint()
        if step >= args.steps:
            break
    if step % args.eval_every != 0:
        evaluate("final")
        checkpoint()
    rates = {f"{s}/{l}": round(aug_tally[(s, l)] / max(1, tot_tally[(s, l)]), 3)
             for (s, l) in tot_tally}
    jlog(event="aug_rates", rates=rates, seconds=round(time.time() - t0))

    # --- export (deployment transform on bundle clips, fp32, 8 s fusion cap) ---
    net.eval()

    def deploy(x):
        return bundle_transform(x, DEPLOY_MAX_S)

    def deploy14(x):
        return bundle_transform(x, 14.0)

    if fold is not None:
        vr = validation_rows(m, fold)
        files = [str(args.bundle / "core" / f) for f in vr.file]
        lg = score_rows(net, files, deploy, device)
        export(vr.path, vr.fold, "inner_oof", lg).to_csv(out / f"scores_{fold}.csv", index=False)
        rep = report((vr.label == "spoof").to_numpy(int), lg)
        jlog(event="export_oof", fold=fold, n=len(vr), **rep)
    else:
        hr = holdout_rows(m)
        files = [str(args.bundle / "core" / f) for f in hr.file]
        lg = score_rows(net, files, deploy, device)
        export(hr.path, "holdout", "holdout", lg).to_csv(out / "scores_holdout.csv", index=False)
        hy = (hr.label == "spoof").to_numpy(int)
        jlog(event="export_holdout", n=len(hr), **report(hy, lg))
        tm = pd.read_csv(args.bundle / "test" / "manifest.csv")
        tfiles = [str(args.bundle / "test" / f) for f in tm.file]
        tl = score_rows(net, tfiles, deploy, device)
        export(tm.path, "test", "test", tl).to_csv(out / "scores_test.csv", index=False)
        jlog(event="export_test", n=len(tm), score_mean=round(float(np.mean(sigmoid(tl))), 4),
             frac_above_half=round(float(np.mean(sigmoid(tl) > 0.5)), 4))
        # diagnostic: whole clips up to 14 s (the >8 s tail), holdout and test
        lg14 = score_rows(net, files, deploy14, device)
        export(hr.path, "holdout", "holdout", lg14).to_csv(out / "diag_holdout_14s.csv", index=False)
        tl14 = score_rows(net, tfiles, deploy14, device)
        export(tm.path, "test", "test", tl14).to_csv(out / "diag_test_14s.csv", index=False)
        jlog(event="diag_14s", holdout=report(hy, lg14),
             test_max_abs_logit_delta_vs_8s=round(float(np.nanmax(np.abs(tl14 - tl))), 4))
        # diagnostics: test-length crops and the fixed augmented slice on the holdout
        hl = LengthMixer(4321, p_test=1.0)
        crops_h = [hl.draw() for _ in range(len(hr))]
        lg_c = np.full(len(hr), np.nan)
        for b in range(0, len(hr), 16):
            ii = list(range(b, min(b + 16, len(hr))))
            clips = [band_match(crop(read_clip(files[j]), crops_h[j],
                                     np.random.default_rng([77, j]))) for j in ii]
            xs, mask = collate(clips)
            lg_c[ii] = score_batch(net, xs, mask, device).numpy()
        pd.DataFrame({"path": hr.path, "crop_s": crops_h, "logit": lg_c}).to_csv(
            out / "diag_holdout_testlen.csv", index=False)
        jlog(event="diag_holdout_testlen", **report(hy, lg_c))
        plans = holdout_slice(len(hr))
        augh = Augmenter(p_aug=1.0)

        def aug_transform_factory(j):
            def f(x):
                x = augh.apply(x, plans[j], np.random.default_rng([55, j]))
                return band_match(x)[: int(DEPLOY_MAX_S * SR)]
            return f

        lg_a = np.full(len(hr), np.nan)
        for b in range(0, len(hr), 16):
            ii = list(range(b, min(b + 16, len(hr))))
            clips = [aug_transform_factory(j)(read_clip(files[j])) for j in ii]
            xs, mask = collate(clips)
            lg_a[ii] = score_batch(net, xs, mask, device).numpy()
        pd.DataFrame({"path": hr.path, "op": [p[0][0] for p in plans], "logit": lg_a}).to_csv(
            out / "diag_holdout_aug.csv", index=False)
        by_op = {}
        for op in sorted({p[0][0] for p in plans}):
            mk = np.array([p[0][0] == op for p in plans])
            by_op[op] = report(hy[mk], lg_a[mk]) if len(set(hy[mk])) == 2 else None
        jlog(event="diag_holdout_aug", overall=report(hy, lg_a), by_op=by_op)

    hashes = save_m5(net, out / "model", {"bundle_tree": meta["tree_sha"], "fold": args.fold,
                                          "arm": args.arm, "steps": step})
    run_meta = {"fold": args.fold, "arm": args.arm, "steps": step, "config": asdict(cfg),
                "config_hash": cfg.hash(), "bundle_tree": meta["tree_sha"],
                "git_sha": meta.get("git_sha"), "device": device,
                "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
                "seconds": round(time.time() - t0), "clips_seen": clips_seen,
                "aug_rates": rates, "shortcut_gate": gate, "hashes": hashes,
                "n_train": len(tr), "val": va_note}  # fmt: skip
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    if args.r2_prefix:
        rclone_push(out, args.r2_prefix.rstrip("/"))
    jlog(event="DONE", seconds=round(time.time() - t0))
    print("DONE")


if __name__ == "__main__":
    main()
