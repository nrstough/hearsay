"""AntiDeepfake backbones (nii-yamagishilab) as off-the-shelf scorers, V2-53.

The released checkpoints are safetensors whose tensor names follow **fairseq's** wav2vec2
module layout (`m_ssl.model.*`) plus a 2-class head (`proj_fc`). fairseq 0.12 does not
install on Python 3.12 (our pin), so this is a **strict transformers port**: every tensor is
classified and every shape checked.

Tensor budget, asserted on every load (run spec `docs/specs/2026-10-11_v2-backbone.md`):

    len(mapped) == expected_mapped(n_layers, n_conv)
    422 (24 layers) · 806 (48 layers) · and the 7 declared-unused and 2 head tensors by name

The load-bearing check is the **mapped count against the architecture**, plus the by-name
declared-unused and head sets. `len(mapped) + len(unused) + len(head) + len(unclassified)`
equals the checkpoint size identically for a dict's keys, so that sum is a tautology rather
than an assertion; it is kept as cheap defence in depth, not relied upon. Target-parameter
coverage is likewise implied once the count matches, nothing is unclassified and the map is
injective -- all three of which are enforced above it. `load_state_dict`
runs with `strict=True`: under `strict=False` a missing tensor leaves that module at its
random initialisation and returns silently.

Why the budget is not enough on its own: all 24 (or 48) encoder layers are shape-identical,
so a reversed or off-by-one layer order loads with zero missing and zero unexpected keys.
`q_proj`/`k_proj`/`v_proj`/`out_proj` are all `[H, H]`, and 14 distinct roles share shape
`(H,)`. Counts and shapes cannot see any of that; only `reference_forward` can, which is why
it exists (test `K7`).

**Positional convolution.** The checkpoint carries the legacy weight-norm names
`pos_conv.0.weight_g` / `weight_v`; torch 2.14 names them
`parametrizations.weight.original0` / `original1`. We emit the legacy names and let torch's
`_load_state_dict_pre_hook` remap them -- verified to work at `strict=True`, including
against the local `weights/wav2vec2-xls-r-300m` checkpoint, which stores the same legacy
names and no `parametrizations.*`.

**Pooling** is parameter-free and therefore not recoverable from the checkpoint. It is
declared here as `POOLING`, from the model card and the winning team's write-up
(`outputs/v2/research/other-teams/files/kevinharvey2025_HactGT13AudioAuthentication__docs_APPROACH.md`
line 279: "adding mean pooling and a linear head with logits [fake, real]"; lines 77 and 537:
"time-mean of the last layer (the input of its linear head)"). Run spec `D6`.

**Direction.** The card orders the head's logits `[fake, real]`, so
`synth_logit = logits[0] - logits[1]` increases with synthetic likelihood. That ordering is
**verified on labelled rows by the scorer's direction gate, never assumed**, and nothing is
ever flipped silently (the M3 precedent, `scripts/score_spectra.py`).

Licence: the AntiDeepfake weights are CC BY-NC-SA 4.0 (non-commercial, share-alike).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

REPO = Path(__file__).resolve().parents[2]

#: How the pooled vector that feeds `proj_fc` is formed. Run spec D6; see module docstring.
POOLING = "time_mean_last_layer"

#: Prefix of every base-SSL tensor in an AntiDeepfake checkpoint.
PREFIX = "m_ssl.model."

#: The 2-class head.
HEAD = ("proj_fc.weight", "proj_fc.bias")

#: Pretraining-only tensors with no inference role (contrastive objective). Declared unused
#: **by name**: an addition or a removal raises rather than being silently dropped.
UNUSED = frozenset(
    {
        f"{PREFIX}quantizer.vars",
        f"{PREFIX}quantizer.weight_proj.weight",
        f"{PREFIX}quantizer.weight_proj.bias",
        f"{PREFIX}project_q.weight",
        f"{PREFIX}project_q.bias",
        f"{PREFIX}final_proj.weight",
        f"{PREFIX}final_proj.bias",
    }
)

#: Non-indexed tensors. Targets are `Wav2Vec2Model` parameter names (no `wav2vec2.` prefix).
STATIC: dict[str, str] = {
    f"{PREFIX}post_extract_proj.weight": "feature_projection.projection.weight",
    f"{PREFIX}post_extract_proj.bias": "feature_projection.projection.bias",
    f"{PREFIX}layer_norm.weight": "feature_projection.layer_norm.weight",
    f"{PREFIX}layer_norm.bias": "feature_projection.layer_norm.bias",
    f"{PREFIX}encoder.layer_norm.weight": "encoder.layer_norm.weight",
    f"{PREFIX}encoder.layer_norm.bias": "encoder.layer_norm.bias",
    f"{PREFIX}encoder.pos_conv.0.bias": "encoder.pos_conv_embed.conv.bias",
    # legacy weight-norm names on purpose; torch's pre-hook remaps them (see docstring)
    f"{PREFIX}encoder.pos_conv.0.weight_g": "encoder.pos_conv_embed.conv.weight_g",
    f"{PREFIX}encoder.pos_conv.0.weight_v": "encoder.pos_conv_embed.conv.weight_v",
    f"{PREFIX}mask_emb": "masked_spec_embed",
}

#: Per-encoder-layer submodule renames, fairseq -> transformers.
LAYER: dict[str, str] = {
    "self_attn.q_proj": "attention.q_proj",
    "self_attn.k_proj": "attention.k_proj",
    "self_attn.v_proj": "attention.v_proj",
    "self_attn.out_proj": "attention.out_proj",
    "self_attn_layer_norm": "layer_norm",
    "fc1": "feed_forward.intermediate_dense",
    "fc2": "feed_forward.output_dense",
    "final_layer_norm": "final_layer_norm",
}

# The index is anchored as (0|[1-9]\d*): a bare \d+ accepts "layers.007", which would map to
# the same target as "layers.7" and let the later write win silently while every count
# still balanced. `invert_map` is what catches the collision; this stops it arising.
_IDX = r"(0|[1-9]\d*)"
_LAYER_RE = re.compile(rf"^{re.escape(PREFIX)}encoder\.layers\.{_IDX}\.(.+)\.(weight|bias)$")
_CONV_RE = re.compile(
    rf"^{re.escape(PREFIX)}feature_extractor\.conv_layers\.{_IDX}\.(0|2\.1)\.(weight|bias)$"
)


@dataclass(frozen=True)
class KeyMapping:
    """Result of classifying every tensor name in a checkpoint.

    `mapped` is checkpoint name -> `Wav2Vec2Model` parameter name. `unused` and `head` are
    checkpoint names. `unclassified` is anything the map does not account for; a non-empty
    `unclassified` is a hard error (test `K5`).
    """

    mapped: dict[str, str]
    unused: frozenset[str]
    head: tuple[str, ...]
    unclassified: frozenset[str]

    @property
    def total(self) -> int:
        return len(self.mapped) + len(self.unused) + len(self.head) + len(self.unclassified)


@contextmanager
def mps_reservation(
    lock_path: Path | str, scope: str, wait_s: float = 0.0, stale_s: float = 43200.0,
) -> Iterator[Path]:  # fmt: skip
    """Exclusive claim on the shared graphics device, with an explicit wait-or-abort path.

    The MPS working set (`recommended_max_memory()`, 13.32 GiB on this machine) is a **device**
    property, shared across every process on the box. Two concurrent 2B jobs total 18.4 GiB and
    both fall off a 120x cliff that no `empty_cache()` recovers from -- and this repo currently
    has three lanes in one checkout, one of them in a sibling worktree.

    Raises `TimeoutError` rather than proceeding.

    Three properties the first version lacked, each of which let two lanes share the device:

    * **A stale lock is only reclaimed if its owner is gone.** It reclaimed purely on age, so a
      second claimant took the lock out from under a live holder still inside its `with` block.
      `os.kill(pid, 0)` decides.
    * **Release only unlinks our own lock.** It unlinked whatever was there, so after a reclaim
      our exit deleted the *other* job's live lock -- two lanes ping-ponging the device.
      A random token written on acquire is checked before unlinking.
    * **`stale_s` defaults to 12 h**, not 2. Stage 2's own forward budget is ~5.8 h, so a 2 h
      window let a sibling lane seize the device in the middle of a legitimate run -- exactly
      the condition this exists to prevent.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(8)
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}\ntoken={token}\nscope={scope}\nacquired={time.time():.0f}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                held = lock_path.read_text()
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue  # released in the window between the failed create and the read
            # Three states, not two. A **live** owner is never reclaimed from, at any age --
            # that is the condition the reservation exists to prevent. A **dead** owner is
            # reclaimed immediately, because its lock is never worth waiting for; gating that
            # behind `age > stale_s` left a killed job's lock blocking the device for the full
            # 12-hour default, which it then did to a real run. Age decides only when liveness
            # is **unknown** (an unparseable or pid-less lock file).
            alive = _owner_alive(held)
            if alive is False or (alive is None and age > stale_s):
                reason = "owner gone" if alive is False else f"owner unknown, stale after {age:.0f}s"
                print(f"reclaiming {lock_path.name} ({reason})")
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{lock_path} held for {age:.0f}s by another job; refusing to share the MPS "
                    f"device (scope wanted: {scope})"
                ) from None
            time.sleep(0.25)
    try:
        yield lock_path
    finally:
        try:
            if f"token={token}" in lock_path.read_text():
                lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def _owner_alive(contents: str) -> bool | None:
    """Is the pid in a lock file still running? `None` when it cannot be determined.

    Tri-state deliberately: "unknown" must not be collapsed into either "alive" (which would
    let a crashed job wedge the device forever) or "dead" (which would let us steal a live
    owner's lock whenever the file was written by something we cannot parse).
    """
    for line in contents.splitlines():
        if line.startswith("pid="):
            try:
                pid = int(line[4:])
            except ValueError:
                return None
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True  # it exists, owned by another user
    return None


def receptive_field(conv_kernel, conv_stride) -> int:
    """Samples needed for one encoder frame: the conv front end's receptive field.

    For the shipped configs (kernels 10,3,3,3,3,2,2 / strides 5,2,2,2,2,2,2) this is **400
    samples, 25 ms**. Below it both the model and `reference_forward` raised an opaque
    `RuntimeError: Kernel size can't be greater than actual input size`, which is a bad answer
    to a legitimate question about a very short clip. The declared policy is to refuse with a
    named error (run spec `I5`/`I6`, Codex C4 amendment).
    """
    r = 1
    for k, s in zip(reversed(list(conv_kernel)), reversed(list(conv_stride)), strict=True):
        r = (r - 1) * s + k
    return r


def check_min_length(cfg: Wav2Vec2Config, n_samples: int) -> None:
    """Refuse a clip shorter than one encoder frame, naming the limit."""
    need = receptive_field(cfg.conv_kernel, cfg.conv_stride)
    if n_samples < need:
        raise ValueError(
            f"{n_samples} samples is below the {need}-sample ({need / 16000 * 1000:.0f} ms) "
            "convolutional receptive field; too short to produce a single encoder frame"
        )


def expected_mapped(n_layers: int, n_conv: int = 7) -> int:
    """How many base-SSL tensors a checkpoint of this shape must carry.

    Derived from the architecture rather than hardcoded, so the count is checked for every
    size rather than only the two we happen to download: 24 layers -> 422, 48 -> 806, and a
    toy 2-layer config -> 70, which is what lets the hermetic tests use the real loader.
    """
    return len(STATIC) + len(LAYER) * 2 * n_layers + 4 * n_conv


def map_keys(ckpt_names, n_layers: int, n_conv: int = 7) -> KeyMapping:
    """Classify every checkpoint tensor name as mapped / unused / head / unclassified.

    Pure name arithmetic: no tensors, no model, no files. `n_layers` and `n_conv` bound the
    indexed families so an out-of-range layer index is unclassified rather than silently
    accepted -- an off-by-one in the checkpoint must not be absorbed by a permissive regex.
    """
    mapped: dict[str, str] = {}
    unused: set[str] = set()
    head: list[str] = []
    unclassified: set[str] = set()

    for name in ckpt_names:
        if name in UNUSED:
            unused.add(name)
            continue
        if name in HEAD:
            head.append(name)
            continue
        if name in STATIC:
            mapped[name] = STATIC[name]
            continue
        m = _LAYER_RE.match(name)
        if m:
            idx, sub, kind = int(m.group(1)), m.group(2), m.group(3)
            if 0 <= idx < n_layers and sub in LAYER:
                mapped[name] = f"encoder.layers.{idx}.{LAYER[sub]}.{kind}"
                continue
        m = _CONV_RE.match(name)
        if m:
            idx, which, kind = int(m.group(1)), m.group(2), m.group(3)
            if 0 <= idx < n_conv:
                leaf = "conv" if which == "0" else "layer_norm"
                mapped[name] = f"feature_extractor.conv_layers.{idx}.{leaf}.{kind}"
                continue
        unclassified.add(name)

    return KeyMapping(mapped, frozenset(unused), tuple(sorted(head)), frozenset(unclassified))


def invert_map(mapped: dict[str, str]) -> dict[str, str]:
    """Target -> checkpoint name. Raises if the forward map is not injective (test `K16`).

    A non-injective map means two checkpoint tensors claim one parameter; the later write
    would win silently and the counts would still balance.
    """
    out: dict[str, str] = {}
    for src, dst in mapped.items():
        if dst in out:
            raise ValueError(f"key map is not injective: {dst!r} claimed by {out[dst]!r} and {src!r}")
        out[dst] = src
    return out


def load_config(weights_dir: str | Path) -> Wav2Vec2Config:
    """Read the shipped `config.json`. Architecture flags are never defaulted (run spec D2).

    `do_stable_layer_norm` decides whether the encoder's final LayerNorm runs before or after
    the layer stack. A wrong value loads every tensor correctly and still computes the wrong
    function, so it is read from the file and asserted present (test `K13`).
    """
    path = Path(weights_dir) / "config.json"
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(raw, dict):
        # ValueError, not TypeError: the bad value is the file's contents, not a caller's
        # argument type. Every other refusal in this function is a ValueError naming the path.
        raise ValueError(  # noqa: TRY004
            f"{path}: expected a JSON object, got {type(raw).__name__}"
        )
    for flag in ("do_stable_layer_norm", "feat_extract_norm", "conv_bias", "num_hidden_layers"):
        if flag not in raw:
            raise ValueError(f"{path}: required architecture flag {flag!r} absent; refusing to default it")
    # F7: assert the VALUES, not merely their presence. A config with
    # do_stable_layer_norm=False loads every tensor correctly and computes a different
    # function; `reference_forward` refuses those flags but is not on the production path.
    if raw["do_stable_layer_norm"] is not True:
        raise ValueError(f"{path}: do_stable_layer_norm={raw['do_stable_layer_norm']!r}; this port covers True only")
    if raw["feat_extract_norm"] != "layer":
        raise ValueError(f"{path}: feat_extract_norm={raw['feat_extract_norm']!r}; this port covers 'layer' only")
    if raw["conv_bias"] is not True:
        raise ValueError(f"{path}: conv_bias={raw['conv_bias']!r}; this port covers True only")
    return Wav2Vec2Config(**raw)


class AntiDeepfakeDetector(nn.Module):
    """A ported AntiDeepfake checkpoint: `Wav2Vec2Model` plus the 2-class linear head.

    `forward` takes `(batch, samples)` of standardized 16 kHz mono audio and returns
    `(batch, 2)` logits ordered `[fake, real]` per the model card. Use `synth_logit` to
    reduce them to one synthetic-likelihood margin.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.head = nn.Linear(config.hidden_size, 2)
        self.pooling = POOLING

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_min_length(self.wav2vec2.config, x.shape[-1])
        hidden = self.wav2vec2(x).last_hidden_state  # (B, T, H)
        if self.pooling != "time_mean_last_layer":  # pragma: no cover - guarded constant
            raise ValueError(f"unsupported pooling {self.pooling!r}")
        return self.head(hidden.mean(dim=1))


def _read_state_dict(weights_dir: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = Path(weights_dir) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"{path} absent; AntiDeepfake checkpoints ship as model.safetensors")
    return load_file(str(path))


def file_sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def load_antideepfake(
    weights_dir: str | Path, device: str | None = None, dtype: torch.dtype = torch.float32,
    identity: dict | None = None,
) -> AntiDeepfakeDetector:  # fmt: skip
    """Load a checkpoint under the strict map, or raise.

    Every claim the module docstring makes is enforced *here*, on the production path. An
    earlier revision asserted the tensor budget only in a `needs_weights` test, so the
    docstring's "asserted on every load" was false on any machine without the download, and
    all four structural guards could be deleted with the whole suite still green.

    fp32 by default (run spec `D7`, amended): the naive MPS fp16 path crosses the device's
    13.32 GiB working set and costs 74 s per forward instead of 0.58 s, permanently for the
    process. If fp16 is wanted, cast on **CPU** and then move to the device.
    """
    weights_dir = Path(weights_dir)
    if identity is not None:
        # A1/A6 at load time, against the bytes on disk. Fetch-time hashing alone let a
        # structurally valid replacement checkpoint load under a stale identity record, because
        # `check_identity` had no production caller and only compared supplied dictionaries.
        #
        # **Every recorded file, not just the weights.** Hashing `model.safetensors` alone left
        # `config.json` unverified, and a shape-compatible config change (a different
        # `do_stable_layer_norm`, a different layer count) alters inference while the checkpoint
        # hash still matches.
        recorded = identity.get("file_sha256") or {}
        if "model.safetensors" not in recorded:
            raise ValueError("identity record carries no sha256 for model.safetensors")
        if "config.json" not in recorded:
            raise ValueError("identity record carries no sha256 for config.json; the architecture is unverified")
        for fname, want in sorted(recorded.items()):
            path = weights_dir / fname
            if not path.exists():
                raise ValueError(f"{weights_dir.name}: {fname} is in the identity record but absent on disk")
            got = file_sha256(path)
            if got != want:
                raise ValueError(
                    f"{weights_dir.name}: {fname} sha256 {got[:16]} does not match the identity "
                    f"record's {want[:16]}; refusing to load an unrecorded checkpoint"
                )
        for field in ("model_id", "revision"):
            if not identity.get(field):
                raise ValueError(f"identity record has no {field!r}; refusing an unpinned load")
    cfg = load_config(weights_dir)
    sd = _read_state_dict(weights_dir)
    n_expected_base = expected_mapped(cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    km = map_keys(sd.keys(), n_layers=cfg.num_hidden_layers, n_conv=cfg.num_feat_extract_layers)

    if km.unclassified:
        raise ValueError(
            f"{weights_dir.name}: {len(km.unclassified)} unclassified tensor(s), refusing to load: "
            f"{sorted(km.unclassified)[:5]}"
        )
    if km.unused != UNUSED:
        missing, extra = sorted(UNUSED - km.unused), sorted(km.unused - UNUSED)
        raise ValueError(
            f"{weights_dir.name}: declared-unused set changed; absent={missing} unexpected={extra}"
        )
    if km.head != tuple(sorted(HEAD)):
        raise ValueError(f"{weights_dir.name}: head tensors {km.head} != {tuple(sorted(HEAD))}")

    # The tensor budget, asserted on the production path (not only in a download-gated test).
    budget = len(km.mapped) + len(km.unused) + len(km.head)
    if budget != len(sd):
        raise ValueError(f"{weights_dir.name}: budget {budget} != {len(sd)} tensors in the checkpoint")
    if len(km.mapped) != n_expected_base:
        raise ValueError(
            f"{weights_dir.name}: {len(km.mapped)} mapped tensors, expected {n_expected_base} "
            f"for {cfg.num_hidden_layers} layers and {cfg.num_feat_extract_layers} conv blocks"
        )
    # Injectivity, enforced rather than merely testable: `renamed` below is a dict, so two
    # sources claiming one target would be a silent overwrite with the counts still balancing.
    invert_map(km.mapped)

    model = AntiDeepfakeDetector(cfg)
    renamed = {km.mapped[k]: sd[k] for k in km.mapped}

    expected = {n for n, _ in model.wav2vec2.named_parameters()}
    # the legacy weight-norm names we emit are remapped by torch's pre-hook, so they are not
    # in named_parameters(); account for them explicitly before comparing coverage.
    legacy = {
        "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    }
    covered = {legacy.get(k, k) for k in renamed}
    if missing := sorted(expected - covered):
        raise ValueError(f"{weights_dir.name}: {len(missing)} target parameter(s) unmapped: {missing[:5]}")

    model.wav2vec2.load_state_dict(renamed, strict=True)

    # The head is loaded through `load_state_dict`, not by assigning `.data`. Assignment
    # bypasses shape validation: a 3-class `proj_fc` landed in an `nn.Linear(H, 2)` leaving
    # `out_features == 2` beside a (3, H) weight, so `forward` returned (B, 3) and broke the
    # [fake, real] contract before any direction gate could see it.
    want_w, want_b = (2, cfg.hidden_size), (2,)
    got_w, got_b = tuple(sd["proj_fc.weight"].shape), tuple(sd["proj_fc.bias"].shape)
    if got_w != want_w or got_b != want_b:
        raise ValueError(
            f"{weights_dir.name}: head shapes {got_w}/{got_b} != {want_w}/{want_b}; "
            "the 2-class [fake, real] head is part of the contract"
        )
    model.head.load_state_dict({"weight": sd["proj_fc.weight"], "bias": sd["proj_fc.bias"]}, strict=True)

    # An implicit cast is refused rather than performed. The checkpoints ship fp32; silently
    # honouring a request that differs from the file hid which precision actually ran.
    # Every float tensor, not a sampled one: an fp16 base with an fp32 head once loaded clean
    # as fp32, silently upcasting the base -- and the harmful orientation silently downcasts it.
    on_disk = {v.dtype for v in sd.values() if v.is_floating_point()}
    if len(on_disk) != 1:
        raise ValueError(f"{weights_dir.name}: mixed tensor dtypes {sorted(map(str, on_disk))}; refusing to guess")
    if {dtype} != on_disk:
        raise ValueError(
            f"{weights_dir.name}: checkpoint is {next(iter(on_disk))}, {dtype} requested. Cast "
            "explicitly with `.to(dtype)` on CPU before moving to the device -- casting on MPS "
            "crosses the 13.32 GiB working set and costs 74 s per forward."
        )
    # The cast is explicit and matches the file, having just been checked against it.
    model = model.to(dtype=dtype)
    # `None` means the repo's default, which is MPS when available -- matching
    # `hearsay.embed.default_device()`. Treating `None` as "leave it on CPU" silently ran a
    # multi-hour scoring job at ~10x the intended wall clock, at 136% CPU and no GPU at all.
    # Pass "cpu" explicitly to mean CPU; the equivalence tests do.
    from hearsay.embed import default_device

    model = model.to(device or default_device())
    return model.eval()


def synth_logit(logits) -> float:
    """`logits[0] - logits[1]`, i.e. fake minus real: increases with synthetic likelihood.

    Verified by the scorer's direction gate on labelled rows before any export is written.
    """
    a = np.asarray(logits, dtype=np.float64).reshape(-1)
    if a.size != 2:
        raise ValueError(f"expected 2 logits, got {a.size}")
    # Same NaN-swallowing class as `max_rel_deviation` had: a non-finite margin that is
    # returned rather than raised becomes a score, and a score becomes a readout.
    if not np.isfinite(a).all():
        raise ValueError(f"non-finite logits {a.tolist()}; refusing to form a margin")
    return float(a[0] - a[1])


#: Fields an identity record must carry before any score file may be bound to it.
IDENTITY_FIELDS = ("model_id", "revision", "licence", "realpath", "file_sha256")


def check_identity(record: dict, loaded: dict) -> None:
    """Bind a score file to the checkpoint that produced it, or raise (`A6`).

    This lived in the test file in the first revision, so the seven Stage-A tests exercised a
    reimplementation and the shipped path had none. The failure it prevents is the one the
    M4/M5 lane hit: a rank reference built from one model and scored with another is silently
    wrong, and nothing downstream can tell.

    **Empty is not a match.** An earlier version compared values only, so two records agreeing
    that a field is `""` or `{}` passed — which is precisely the stale-record case this is for.
    """
    for field in IDENTITY_FIELDS:
        if field not in record:
            raise ValueError(f"identity record missing {field!r}")
        if not record[field]:
            raise ValueError(f"identity record has an empty {field!r}; refusing to treat it as a match")
        if record[field] != loaded.get(field):
            raise ValueError(
                f"identity mismatch on {field!r}: recorded {record[field]!r} vs loaded {loaded.get(field)!r}"
            )


# --- independent reference implementation (test K7) -----------------------------------------
#
# Written from the fairseq-named tensors directly, with no use of the key map or of
# Wav2Vec2Model. Agreement between this and the ported model is the only check that can catch
# a shape-compatible mis-map: a q/k swap, a swap among the 14 roles sharing shape (H,), or a
# reversed encoder-layer order, all of which load with zero missing and zero unexpected keys.
#
# Run it at fp32 on CPU. At fp16 the numerical floor (max|d| 0.127 on a real clip) is only 5x
# below a q/k swap's signature (0.646), so a subtler single-tensor swap hides inside it.


def _weight_norm_weight(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reconstruct a weight-normalised conv kernel: g * v / ||v||, norm over dims != 2.

    The norm axis is the whole content of the pos_conv claim. Taking it over dim 2 instead
    changes the result by O(1) on an O(1) tensor, so this is not a near-miss if wrong.
    """
    return g * v / v.pow(2).sum(dim=(0, 1), keepdim=True).sqrt()


@torch.inference_mode()
def reference_forward(sd: dict[str, torch.Tensor], cfg: Wav2Vec2Config, x: torch.Tensor) -> torch.Tensor:
    """`(batch, 2)` logits for standardized audio `(batch, samples)`, from fairseq names only.

    Mirrors `Wav2Vec2Model` with `do_stable_layer_norm=True` and `feat_extract_norm="layer"`:
    conv front end (conv -> LayerNorm over channels -> GELU) -> feature projection
    (LayerNorm -> linear) -> positional conv added -> pre-norm transformer layers ->
    final encoder LayerNorm -> time mean -> linear head.
    """
    if not cfg.do_stable_layer_norm or cfg.feat_extract_norm != "layer":
        raise ValueError("reference_forward covers do_stable_layer_norm=True, feat_extract_norm='layer' only")
    check_min_length(cfg, x.shape[-1])

    h = x.unsqueeze(1) if x.dim() == 2 else x  # (B, 1, S)
    for i, (k, s) in enumerate(zip(cfg.conv_kernel, cfg.conv_stride, strict=True)):
        w = sd[f"{PREFIX}feature_extractor.conv_layers.{i}.0.weight"]
        # F13: `sd[...]`, not `sd.get(...)`. A silently dropped bias returned a finite, wrong
        # answer from the one function whose whole purpose is to be the independent check.
        bias_key = f"{PREFIX}feature_extractor.conv_layers.{i}.0.bias"
        if cfg.conv_bias:
            b = sd[bias_key]
        else:
            if bias_key in sd:
                raise ValueError(f"conv_bias is False but {bias_key} is present")
            b = None
        h = F.conv1d(h, w, b, stride=s)
        ln_w = sd[f"{PREFIX}feature_extractor.conv_layers.{i}.2.1.weight"]
        ln_b = sd[f"{PREFIX}feature_extractor.conv_layers.{i}.2.1.bias"]
        h = F.layer_norm(h.transpose(-2, -1), (ln_w.shape[0],), ln_w, ln_b).transpose(-2, -1)
        h = F.gelu(h)

    h = h.transpose(1, 2)  # (B, T, 512)
    h = F.layer_norm(h, (h.shape[-1],), sd[f"{PREFIX}layer_norm.weight"], sd[f"{PREFIX}layer_norm.bias"])
    h = F.linear(h, sd[f"{PREFIX}post_extract_proj.weight"], sd[f"{PREFIX}post_extract_proj.bias"])

    pos_w = _weight_norm_weight(
        sd[f"{PREFIX}encoder.pos_conv.0.weight_g"], sd[f"{PREFIX}encoder.pos_conv.0.weight_v"]
    )
    k = cfg.num_conv_pos_embeddings
    p = F.conv1d(
        h.transpose(1, 2), pos_w, sd[f"{PREFIX}encoder.pos_conv.0.bias"],
        padding=k // 2, groups=cfg.num_conv_pos_embedding_groups,
    )  # fmt: skip
    if k % 2 == 0:  # Wav2Vec2SamePadLayer trims one frame for an even kernel
        p = p[..., :-1]
    h = h + F.gelu(p).transpose(1, 2)

    heads, dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    for i in range(cfg.num_hidden_layers):
        pre = f"{PREFIX}encoder.layers.{i}."
        res = h
        h = F.layer_norm(
            h, (cfg.hidden_size,), sd[f"{pre}self_attn_layer_norm.weight"], sd[f"{pre}self_attn_layer_norm.bias"]
        )
        b_, t_, _ = h.shape

        def proj(name, z=h, pre=pre, b_=b_, t_=t_):
            w = F.linear(z, sd[f"{pre}self_attn.{name}.weight"], sd[f"{pre}self_attn.{name}.bias"])
            return w.view(b_, t_, heads, dim).transpose(1, 2)

        a = F.scaled_dot_product_attention(proj("q_proj"), proj("k_proj"), proj("v_proj"))
        a = a.transpose(1, 2).reshape(b_, t_, cfg.hidden_size)
        h = res + F.linear(a, sd[f"{pre}self_attn.out_proj.weight"], sd[f"{pre}self_attn.out_proj.bias"])

        res = h
        f = F.layer_norm(
            h, (cfg.hidden_size,), sd[f"{pre}final_layer_norm.weight"], sd[f"{pre}final_layer_norm.bias"]
        )
        f = F.gelu(F.linear(f, sd[f"{pre}fc1.weight"], sd[f"{pre}fc1.bias"]))
        h = res + F.linear(f, sd[f"{pre}fc2.weight"], sd[f"{pre}fc2.bias"])

    h = F.layer_norm(
        h, (cfg.hidden_size,),
        sd[f"{PREFIX}encoder.layer_norm.weight"], sd[f"{PREFIX}encoder.layer_norm.bias"],
    )  # fmt: skip
    return F.linear(h.mean(dim=1), sd["proj_fc.weight"], sd["proj_fc.bias"])


def max_rel_deviation(a: torch.Tensor, b: torch.Tensor) -> float:
    """`max|a - b| / max|b|`. The gate `K7` uses, in place of cosine similarity.

    Cosine is unusable here: a deliberate q/k swap passes `cos > 0.99` at 0.99083, while this
    statistic separates it from the fp32 numerical floor by four orders of magnitude.

    **Non-finite input returns `inf`, never NaN.** A NaN would compare `False` against any
    threshold, so a model whose output went non-finite would pass a `deviation < tol` gate
    silently. Found by the pos_conv negative control, which divides by a zero norm.
    """
    a64, b64 = a.detach().double(), b.detach().double()
    # F14: broadcasting turned a batch-size mismatch into a plausible number -- (1,2) vs
    # (8,2) returned 0.5 and (2,) vs (3,2) returned 0.0 -- and K7/K18 both gate on it.
    if a64.shape != b64.shape:
        raise ValueError(f"shape mismatch: {tuple(a64.shape)} vs {tuple(b64.shape)}; refusing to broadcast")
    if not (torch.isfinite(a64).all() and torch.isfinite(b64).all()):
        return float("inf")
    scale = b64.abs().max().item()
    if scale == 0:
        raise ValueError("reference output is all zeros; cannot form a relative deviation")
    return float((a64 - b64).abs().max().item() / scale)
