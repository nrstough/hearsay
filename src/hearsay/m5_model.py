"""M5 model: a truncated XLS-R 300M with a learned layer-weighted sum, masked attentive
statistics pooling and a linear head. High logit = synthetic (spec D5).

Save/load is a full truncated model (safetensors) plus the head, so the CPU scorer in Docker
needs no surgery: `load_m5(dir)` rebuilds exactly what was trained and refuses on hash drift.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


@dataclass
class M5Config:
    keep_layers: int = 12
    train_top: int = 12  # trainable transformer layers, counted from the top of the kept stack
    layer_weighted_sum: bool = True
    pooling: str = "attn_stats"  # or "mean"
    bottleneck: int = 128
    lr_backbone: float = 1e-5
    llrd: float = 0.85
    lr_head: float = 5e-4
    weight_decay: float = 0.01
    warmup: float = 0.08
    schedule: str = "cosine"
    label_smoothing: float = 0.05
    batch_size: int = 32
    grad_clip: float = 1.0
    layerdrop: float = 0.0
    spec_augment: bool = True
    p_aug: float = 0.65
    p_rawboost: float = 0.25
    p_test_len: float = 0.7
    seed: int = 0
    mix: dict = field(default_factory=lambda: {
        "real": {"ljspeech": 0.25, "librispeech": 0.35, "asvspoof2019": 0.40},
        "spoof": {"diffssd": 0.78, "mlaad": 0.12, "asvspoof2019": 0.10},  # MLAAD 12%: probe A
        "mlaad_cap_per_model": 120,
    })

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


TINY = {"hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 2,
        "intermediate_size": 64, "conv_dim": (32,) * 7, "conv_kernel": (10, 3, 3, 3, 3, 2, 2),
        "conv_stride": (5, 2, 2, 2, 2, 2, 2), "feat_extract_norm": "layer",
        "do_stable_layer_norm": True, "vocab_size": 32, "num_conv_pos_embeddings": 16,
        "num_conv_pos_embedding_groups": 2}  # fmt: skip


class AttnStatsPool(nn.Module):
    """Masked attentive statistics pooling: attention-weighted mean and std over frames."""

    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, bottleneck), nn.Tanh(), nn.Linear(bottleneck, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D); mask: (B, T) with 1 on valid frames
        logits = self.attn(h).squeeze(-1).float()
        logits = logits.masked_fill(mask == 0, -1e4)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, T, 1)
        mean = (w * h.float()).sum(1)
        var = (w * (h.float() - mean.unsqueeze(1)) ** 2).sum(1)
        return torch.cat([mean, torch.sqrt(var + 1e-6)], dim=-1)


class M5Net(nn.Module):
    def __init__(self, backbone: Wav2Vec2Model, cfg: M5Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        n_layers = len(backbone.encoder.layers)
        dim = backbone.config.hidden_size
        # hidden_states has n_layers + 1 entries (the projected input plus every layer)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers + 1)) if cfg.layer_weighted_sum else None
        if cfg.pooling == "attn_stats":
            self.pool = AttnStatsPool(dim, cfg.bottleneck)
            self.head = nn.Linear(2 * dim, 1)
        else:
            self.pool = None
            self.head = nn.Linear(dim, 1)

    def layer_weights(self) -> torch.Tensor | None:
        return None if self.layer_logits is None else torch.softmax(self.layer_logits, 0)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_values, attention_mask=attention_mask, output_hidden_states=True)
        if self.layer_logits is not None:
            hs = torch.stack(out.hidden_states, dim=0)  # (L+1, B, T, D)
            w = torch.softmax(self.layer_logits, 0).to(hs.dtype).view(-1, 1, 1, 1)
            h = (w * hs).sum(0)
        else:
            h = out.last_hidden_state
        fmask = self.backbone._get_feature_vector_attention_mask(h.shape[1], attention_mask)
        if self.pool is not None:
            pooled = self.pool(h, fmask)
        else:
            m = fmask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled.to(self.head.weight.dtype)).squeeze(-1)  # high = synthetic


def build_model(cfg: M5Config, weights_dir: str | Path | None = None, tiny: bool = False) -> M5Net:
    """Load XLS-R (or a tiny random-init twin for tests), truncate, freeze, attach the head."""
    if tiny:
        bb = Wav2Vec2Model(Wav2Vec2Config(**TINY, layerdrop=cfg.layerdrop,
                                          apply_spec_augment=cfg.spec_augment))
    else:
        bb = Wav2Vec2Model.from_pretrained(str(weights_dir), layerdrop=cfg.layerdrop,
                                           apply_spec_augment=cfg.spec_augment)
    keep = min(cfg.keep_layers, len(bb.encoder.layers))
    bb.encoder.layers = bb.encoder.layers[:keep]
    bb.config.num_hidden_layers = keep
    bb.freeze_feature_encoder()
    n_frozen = max(0, keep - cfg.train_top)
    for layer in bb.encoder.layers[:n_frozen]:
        for p in layer.parameters():
            p.requires_grad_(False)
    if n_frozen > 0:  # nothing below the first trainable layer moves either
        for p in bb.feature_projection.parameters():
            p.requires_grad_(False)
        if bb.encoder.pos_conv_embed is not None:
            for p in bb.encoder.pos_conv_embed.parameters():
                p.requires_grad_(False)
    if cfg.train_top == 0:  # the frozen recipe: no backbone parameter moves at all (Codex 3)
        for p in bb.parameters():
            p.requires_grad_(False)
    return M5Net(bb, cfg)


def param_groups(net: M5Net, cfg: M5Config) -> list[dict]:
    """Layer-wise LR decay from the top kept layer downward; head at lr_head, no weight decay;
    no weight decay on biases and LayerNorm weights."""
    groups: list[dict] = []
    layers = list(net.backbone.encoder.layers)
    n = len(layers)
    for i, layer in enumerate(layers):
        depth_from_top = n - 1 - i
        lr = cfg.lr_backbone * (cfg.llrd**depth_from_top)
        decay = [p for nme, p in layer.named_parameters()
                 if p.requires_grad and not (nme.endswith("bias") or "layer_norm" in nme)]
        no_decay = [p for nme, p in layer.named_parameters()
                    if p.requires_grad and (nme.endswith("bias") or "layer_norm" in nme)]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": cfg.weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    rest = [p for nme, p in net.backbone.named_parameters()
            if p.requires_grad and not nme.startswith("encoder.layers.")]
    if rest:  # encoder layer norm etc. at the lowest backbone lr
        groups.append({"params": rest, "lr": cfg.lr_backbone * (cfg.llrd ** (n - 1)),
                       "weight_decay": 0.0})
    head = [p for nme, p in net.named_parameters()
            if p.requires_grad and not nme.startswith("backbone.")]
    groups.append({"params": head, "lr": cfg.lr_head, "weight_decay": 0.0})
    return groups


def trainable_names(net: M5Net) -> list[str]:
    return [n for n, p in net.named_parameters() if p.requires_grad]


def save_m5(net: M5Net, out: str | Path, extra_meta: dict | None = None) -> dict:
    """Full truncated backbone (safetensors) + head/pool/layer weights + config + hashes."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    net.backbone.save_pretrained(out / "backbone", safe_serialization=True)
    head_sd = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()
               if not k.startswith("backbone.")}
    save_file(head_sd, str(out / "head.safetensors"))
    (out / "m5_config.json").write_text(json.dumps(asdict(net.cfg), indent=2))
    hashes = {
        "config_hash": net.cfg.hash(),
        "backbone_sha256": _sha(out / "backbone" / "model.safetensors"),
        "head_sha256": _sha(out / "head.safetensors"),
        **(extra_meta or {}),
    }
    (out / "hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))
    return hashes


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_m5(path: str | Path, device: str = "cpu") -> M5Net:
    """Rebuild a saved model; refuses if any file's sha differs from hashes.json (E6)."""
    from safetensors.torch import load_file

    path = Path(path)
    hashes = json.loads((path / "hashes.json").read_text())
    for key, rel in (("backbone_sha256", "backbone/model.safetensors"),
                     ("head_sha256", "head.safetensors")):
        got = _sha(path / rel)
        if got != hashes[key]:
            raise RuntimeError(f"{rel}: sha {got[:12]} != recorded {hashes[key][:12]}")
    cfg = M5Config(**json.loads((path / "m5_config.json").read_text()))
    if cfg.hash() != hashes["config_hash"]:
        raise RuntimeError("m5_config.json does not match the recorded config hash")
    bb = Wav2Vec2Model.from_pretrained(str(path / "backbone"), layerdrop=0.0,
                                       apply_spec_augment=False)
    cfg_loaded = M5Config(**{**asdict(cfg), "keep_layers": len(bb.encoder.layers)})
    net = M5Net(bb, cfg_loaded)
    net.load_state_dict({**{f"backbone.{k}": v for k, v in bb.state_dict().items()},
                         **load_file(str(path / "head.safetensors"))})
    net.cfg = cfg
    return net.eval().to(device)


@torch.no_grad()
def score_batch(net: M5Net, xs, mask, device: str = "cpu") -> torch.Tensor:
    """Float32 logits for a collated batch (numpy in, tensor out)."""
    x = torch.as_tensor(xs, dtype=torch.float32, device=device)
    m = torch.as_tensor(mask, dtype=torch.long, device=device)
    return net(x, m).float().cpu()


def bce_with_smoothing(logits: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    target = y.float() * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(logits.float(), target)
