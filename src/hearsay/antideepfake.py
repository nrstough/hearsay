"""AntiDeepfake backbones (nii-yamagishilab) as off-the-shelf scorers, V2-53.

The released checkpoints are safetensors whose tensor names follow **fairseq's** wav2vec2
module layout (`m_ssl.model.*`) plus a 2-class head (`proj_fc`). fairseq 0.12 does not
install on Python 3.12 (our pin), so this is a **strict transformers port**: every tensor is
classified and every shape checked.

Tensor budget, asserted on every load (run spec `docs/specs/2026-10-11_v2-backbone.md`):

    len(mapped) + len(UNUSED) + len(HEAD) == len(checkpoint)
    431 == 422 + 7 + 2   (24 layers)        815 == 806 + 7 + 2   (48 layers)

and every parameter of the target `Wav2Vec2Model` must receive a tensor. `load_state_dict`
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

import json
import re
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

_LAYER_RE = re.compile(rf"^{re.escape(PREFIX)}encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$")
_CONV_RE = re.compile(
    rf"^{re.escape(PREFIX)}feature_extractor\.conv_layers\.(\d+)\.(0|2\.1)\.(weight|bias)$"
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
    raw = json.loads(path.read_text())
    for flag in ("do_stable_layer_norm", "feat_extract_norm", "conv_bias", "num_hidden_layers"):
        if flag not in raw:
            raise ValueError(f"{path}: required architecture flag {flag!r} absent; refusing to default it")
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


def load_antideepfake(
    weights_dir: str | Path, device: str | None = None, dtype: torch.dtype = torch.float32,
) -> AntiDeepfakeDetector:  # fmt: skip
    """Load a checkpoint under the strict map, or raise.

    fp32 by default (run spec `D7`, amended): the naive MPS fp16 path crosses the device's
    13.32 GiB working set and costs 74 s per forward instead of 0.58 s, permanently for the
    process. If fp16 is wanted, cast on **CPU** and then move to the device.
    """
    weights_dir = Path(weights_dir)
    cfg = load_config(weights_dir)
    sd = _read_state_dict(weights_dir)
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

    model = AntiDeepfakeDetector(cfg)
    target = {k: v for k, v in sd.items() if k in km.mapped}
    renamed = {km.mapped[k]: v for k, v in target.items()}

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
    model.head.weight.data = sd["proj_fc.weight"].clone()
    model.head.bias.data = sd["proj_fc.bias"].clone()

    model = model.to(dtype=dtype)
    if device is not None:
        model = model.to(device)
    return model.eval()


def synth_logit(logits) -> float:
    """`logits[0] - logits[1]`, i.e. fake minus real: increases with synthetic likelihood.

    Verified by the scorer's direction gate on labelled rows before any export is written.
    """
    a = np.asarray(logits, dtype=np.float64).reshape(-1)
    if a.size != 2:
        raise ValueError(f"expected 2 logits, got {a.size}")
    return float(a[0] - a[1])


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

    h = x.unsqueeze(1) if x.dim() == 2 else x  # (B, 1, S)
    for i, (k, s) in enumerate(zip(cfg.conv_kernel, cfg.conv_stride, strict=True)):
        w = sd[f"{PREFIX}feature_extractor.conv_layers.{i}.0.weight"]
        b = sd.get(f"{PREFIX}feature_extractor.conv_layers.{i}.0.bias") if cfg.conv_bias else None
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
    if not (torch.isfinite(a64).all() and torch.isfinite(b64).all()):
        return float("inf")
    scale = b64.abs().max().item()
    if scale == 0:
        raise ValueError("reference output is all zeros; cannot form a relative deviation")
    return float((a64 - b64).abs().max().item() / scale)
