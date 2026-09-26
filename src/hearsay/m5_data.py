"""M5 data plumbing: the training manifest (fold discipline, extra pools), crops and batching.

Fold discipline (spec D1/D6): the shared fold file is read-only (sha-pinned). Extra rows are
training-only: they never get a score in the fusion export and never enter the holdout. A fold
model never trains on its own validation fold, on any holdout row, or on an extra spoof
generator from the same family as one of its validation generators (FAMILY_EXCLUSIONS).

Crops (spec D3): one crop length per batch, drawn 70/30 from the NSA test durations and a
uniform 3-14 s; a clip shorter than the draw is used whole; padding is zeros plus an attention
mask, never repetition; normalization runs over valid samples only (== normalize_windows on the
unpadded clip).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay import SR
from hearsay.embed import normalize_windows, prepare_segment, test_duration_sampler

REPO = Path(__file__).resolve().parents[2]
FOLDS = REPO / "splits" / "nsa_folds.csv"
FOLDS_PLUS = REPO / "splits" / "nsa_folds_plus_asv19.csv"
FOLDS_SHA256 = "a607d3eccd9a959aea83768ec317502a72b9d242a7efc6c6533336a331ec8037"
FOLDS_PLUS_SHA256 = "7bcd377d5414bb2a9a752d022175ac2d0062fff6abe11ff0f8aa9a4cc5f3e196"

INNER_FOLDS = ("0", "1", "2", "3", "4")
HOLDOUT_GENERATORS = frozenset({"playht", "wavegrad2"})
HOLDOUT_ALIASES = ("playht", "play.ht", "play_ht", "wavegrad")
# fold -> name fragments of extra spoof generators that belong to that fold's validation
# families; a substring match on the lower-cased generator / model_name excludes the row.
FAMILY_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "1": ("openvoice",),
    "2": ("xtts", "vixtts"),
    "4": ("elevenlabs", "eleven_labs", "eleven-labs"),
}
CORE_SOURCES = frozenset({"diffssd", "ljspeech", "librispeech"})
MAX_CROP_S = 14.0
MIN_CROP_S = 3.0
DEPLOY_MAX_S = 8.0  # the fusion cap: prepare_segment's default, shared with M1 and every detector


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_folds_file(path: str | Path = FOLDS, expected: str = FOLDS_SHA256) -> None:
    """Tripwire: the shared fold file must be byte-identical to the one M5 was planned on."""
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"{path} sha256 {got[:12]}... != pinned {expected[:12]}...; "
                           "the shared fold file changed, stop and ask")  # fmt: skip


def group_key(row) -> str:
    """Identical to scripts/make_folds.py:group_key, so extra NSA rows land in the same groups."""
    if row.label == "spoof":
        return f"gen:{row.generator}"
    if row.source == "ljspeech":
        return f"lj:{str(row.utt)[:5]}"
    return f"spk:{row.speaker}"


def _row_id(source: str, path: str) -> str:
    return f"{source}_{hashlib.sha1(path.encode()).hexdigest()[:12]}"


def _has_holdout_alias(names: pd.Series) -> pd.Series:
    low = names.astype(str).str.lower()
    return low.apply(lambda s: any(a in s for a in HOLDOUT_ALIASES))


def build_manifest(
    folds: pd.DataFrame,
    full: pd.DataFrame,
    mlaad: pd.DataFrame | None = None,
    *,
    extra_spoof_cap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Rows: id, path, label, generator, speaker, source, group, fold, train_scope, model_name.

    folds: the (extended) fold file. Rows with a CORE source are `core` (scored, foldable);
    other sources in it (ASV19 from the main chat's extension) are `extra_asv19`: they keep
    their fold (never trained on by that fold's model) but are never exported.
    full: outputs/manifests/nsa_train_full.csv; rows not in `folds` become `extra_nsa`, inherit
    the fold of their group, and are dropped if that fold is the holdout. Extra spoof is capped
    per generator (the validation is generator-grouped; within-generator volume adds little).
    mlaad: spoof-only, fold "extra" (train-only everywhere, minus FAMILY_EXCLUSIONS).
    """
    rng = np.random.default_rng(seed)
    f = folds.copy()
    f["fold"] = f["fold"].astype(str)
    f["train_scope"] = np.where(f.source.isin(CORE_SOURCES), "core", "extra_" + f.source.map(
        lambda s: "asv19" if "asv" in s else s))
    f["model_name"] = f.generator
    parts = [f]

    core_paths = set(f.path)
    group_fold = f[f.train_scope == "core"].drop_duplicates("group").set_index("group").fold
    x = full[~full.path.isin(core_paths)].copy()
    if len(x):
        x["group"] = x.apply(group_key, axis=1)
        unknown = set(x.group) - set(group_fold.index)
        if unknown:
            raise ValueError(f"extra NSA rows in groups absent from the fold file: "
                             f"{sorted(unknown)[:5]}")  # fmt: skip
        x["fold"] = x.group.map(group_fold)
        x = x[x.fold != "holdout"]
        x = x[~x.generator.isin(HOLDOUT_GENERATORS)]
        keep = []
        for g, d in x[x.label == "spoof"].groupby("generator"):
            idx = d.index.to_numpy()
            if len(idx) > extra_spoof_cap:
                idx = rng.choice(idx, size=extra_spoof_cap, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep) if keep else np.array([], dtype=int)
        x = pd.concat([x.loc[np.sort(keep)], x[x.label == "bonafide"]])
        x["train_scope"] = "extra_nsa"
        x["model_name"] = x.generator
        parts.append(x[["path", "label", "generator", "speaker", "source", "group", "fold",
                        "train_scope", "model_name"]])  # fmt: skip

    if mlaad is not None and len(mlaad):
        m = mlaad.drop_duplicates("path").copy()
        if (m.label != "spoof").any():
            raise ValueError("MLAAD rows must all be spoof")
        if _has_holdout_alias(m.model_name).any() or _has_holdout_alias(m.generator).any():
            bad = m[_has_holdout_alias(m.model_name) | _has_holdout_alias(m.generator)]
            raise ValueError(f"extra generator matches a holdout family: "
                             f"{sorted(set(bad.model_name))[:5]}")  # fmt: skip
        m = pd.DataFrame({
            "path": m.path, "label": "spoof", "generator": m.model_name,
            "speaker": m.get("speaker", pd.Series("", index=m.index)),
            "source": "mlaad", "group": "mlaad:" + m.model_name.astype(str),
            "fold": "extra", "train_scope": "extra_mlaad", "model_name": m.model_name,
        })  # fmt: skip
        parts.append(m)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates("path", keep="first").reset_index(drop=True)
    if out.path.str.contains("in_the_wild", case=False).any():
        raise ValueError("In-the-Wild paths are never training data")
    out.insert(0, "id", [_row_id(s, p) for s, p in zip(out.source, out.path)])
    assert out.id.is_unique
    return out


def _excluded_family(m: pd.DataFrame, fold: str) -> pd.Series:
    frags = FAMILY_EXCLUSIONS.get(str(fold), ())
    if not frags:
        return pd.Series(False, index=m.index)
    low = m.model_name.astype(str).str.lower()
    extra_spoof = m.train_scope.str.startswith("extra") & (m.label == "spoof")
    return extra_spoof & low.apply(lambda s: any(fr in s for fr in frags))


def training_rows(m: pd.DataFrame, fold: str | None) -> pd.DataFrame:
    """Rows the model for `fold` trains on (fold=None: the full model, all inner folds).

    Never a holdout row; never the model's own validation fold; never an extra spoof generator
    from the model's validation families.
    """
    r = m[m.fold != "holdout"]
    if fold is not None:
        fold = str(fold)
        r = r[r.fold != fold]
        r = r[~_excluded_family(r, fold)]
    assert not r.fold.eq("holdout").any()
    assert set(r.label) == {"spoof", "bonafide"}, "a training set needs both classes"
    return r


def validation_rows(m: pd.DataFrame, fold: str) -> pd.DataFrame:
    """Core rows of inner fold `fold` (these get the model's out-of-fold scores)."""
    r = m[(m.fold == str(fold)) & (m.train_scope == "core")]
    assert set(r.label) == {"spoof", "bonafide"}
    return r


def holdout_rows(m: pd.DataFrame) -> pd.DataFrame:
    return m[(m.fold == "holdout") & (m.train_scope == "core")]


def shortcut_aucs(rows: pd.DataFrame, features=("duration", "peak", "lead_s")) -> dict:
    """max(AUC, 1-AUC) of each trivial feature against the label, per train_scope (that scope
    vs the opposite class of the whole pool, so a spoof-only pool is still measurable) and
    pooled. Computed on the rows given (a model's training rows), never on holdout rows."""
    from sklearn.metrics import roc_auc_score

    out: dict[str, dict[str, float]] = {}
    y_all = (rows.label == "spoof").to_numpy(int)

    def _auc(mask: np.ndarray, feat: str) -> float | None:
        v = rows[feat].to_numpy(float)
        ok = mask & np.isfinite(v)
        if ok.sum() < 20 or len(set(y_all[ok])) < 2:
            return None
        a = roc_auc_score(y_all[ok], v[ok])
        return round(float(max(a, 1 - a)), 4)

    for scope in sorted(set(rows.train_scope)):
        in_scope = (rows.train_scope == scope).to_numpy()
        scope_labels = set(y_all[in_scope])
        # the scope's rows vs the opposite class(es) drawn from the whole pool
        mask = in_scope | ~np.isin(y_all, list(scope_labels))
        out[scope] = {f: _auc(mask, f) for f in features}
    out["pooled"] = {f: _auc(np.ones(len(rows), bool), f) for f in features}
    return out


# --- crops and batching -------------------------------------------------------------------


class LengthMixer:
    """Per-batch crop length: p_test from the NSA test durations, else uniform [lo, hi]."""

    def __init__(self, seed: int = 0, p_test: float = 0.7, lo: float = MIN_CROP_S,
                 hi: float = MAX_CROP_S):  # fmt: skip
        self.rng = np.random.default_rng(seed)
        self.draw_test = test_duration_sampler(seed)
        self.p_test, self.lo, self.hi = p_test, lo, hi

    def draw(self) -> float:
        if self.rng.random() < self.p_test:
            return float(np.clip(self.draw_test(), self.lo, self.hi))
        return float(self.rng.uniform(self.lo, self.hi))


def crop(x: np.ndarray, crop_s: float, rng: np.random.Generator) -> np.ndarray:
    """Random crop of `crop_s` seconds; a shorter clip is returned whole (no padding here)."""
    n = int(crop_s * SR)
    if x.size <= n:
        return np.ascontiguousarray(x, dtype=np.float32)
    off = int(rng.integers(0, x.size - n + 1))
    return np.ascontiguousarray(x[off : off + n], dtype=np.float32)


def collate(clips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(B, T) float32 normalized over valid samples + zero padding, and a (B, T) int mask."""
    t = max(c.size for c in clips)
    xs = np.zeros((len(clips), t), np.float32)
    mask = np.zeros((len(clips), t), np.int64)
    for i, c in enumerate(clips):
        xs[i, : c.size] = normalize_windows(c[None, :])[0]
        mask[i, : c.size] = 1
    return xs, mask


def deploy_transform(x: np.ndarray, max_s: float = DEPLOY_MAX_S) -> np.ndarray:
    """The deployment transform on RAW audio (Docker, m5_score.py): exactly M1's
    prepare_segment (band-limit at the NSA 7.25 kHz wall -> trim -> cap) then normalize."""
    return normalize_windows(prepare_segment(x, max_s=max_s)[None, :])[0]


def band_match(x: np.ndarray) -> np.ndarray:
    """The deterministic NSA band-limit (hearsay.handcrafted.band_limit), applied to EVERY clip
    the model sees: training crops after augmentation, validation crops, and exports."""
    from hearsay.handcrafted import band_limit

    return band_limit(x)


def bundle_transform(x_flac: np.ndarray, max_s: float = DEPLOY_MAX_S) -> np.ndarray:
    """The deployment transform on a BUNDLE clip (already trim_silence'd at build time):
    band-limit -> cap -> normalize. Differs from deploy_transform only in the order of trim
    and band-limit, which moves the trim boundary by at most a frame (tested)."""
    return normalize_windows(band_match(x_flac)[: int(max_s * SR)][None, :])[0]


def epoch_rng(seed: int, epoch: int, idx: int) -> np.random.Generator:
    """Deterministic per (seed, epoch, row): same crop for the same triple, new crop per epoch."""
    return np.random.default_rng([seed, epoch, idx])
