"""M5 model on a tiny random-init wav2vec2 twin, CPU only (spec appendix E1-E8)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hearsay import SR
from hearsay.m5_data import collate
from hearsay.m5_model import (
    M5Config,
    bce_with_smoothing,
    build_model,
    load_m5,
    param_groups,
    save_m5,
    score_batch,
    trainable_names,
)

torch.set_num_threads(2)


def _batch(seed: int = 0, lengths=(1.0, 0.6)):
    rng = np.random.default_rng(seed)
    clips = [(0.3 * rng.standard_normal(int(s * SR))).astype(np.float32) for s in lengths]
    xs, mask = collate(clips)
    return torch.from_numpy(xs), torch.from_numpy(mask)


def _step(net, cfg, seed=0):
    torch.manual_seed(seed)
    x, m = _batch(seed)
    y = torch.tensor([1, 0])
    opt = torch.optim.AdamW(param_groups(net, cfg))
    loss = bce_with_smoothing(net(x, m), y, cfg.label_smoothing)
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_e1_frozen_and_trainable_sets_with_counterfactual():
    cfg = M5Config(keep_layers=3, train_top=2)
    net = build_model(cfg, tiny=True)
    assert len(net.backbone.encoder.layers) == 3
    names = trainable_names(net)
    assert not any(n.startswith("backbone.feature_extractor") for n in names)
    assert not any(n.startswith("backbone.encoder.layers.0.") for n in names)
    assert any(n.startswith("backbone.encoder.layers.1.") for n in names)
    assert any(n.startswith("backbone.encoder.layers.2.") for n in names)
    assert "head.weight" in names and "layer_logits" in names
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    net.train()
    _step(net, cfg)
    for n, p in net.named_parameters():
        moved = not torch.equal(before[n], p.detach())
        if n.startswith(("backbone.feature_extractor", "backbone.encoder.layers.0.")):
            assert not moved, n
    assert any(not torch.equal(before[n], p.detach())
               for n, p in net.named_parameters() if n.startswith("backbone.encoder.layers.2."))
    # counterfactual: train_top=0 freezes every transformer layer
    net0 = build_model(M5Config(keep_layers=3, train_top=0), tiny=True)
    assert not any(n.startswith("backbone.encoder.layers") for n in trainable_names(net0))


@pytest.mark.parametrize("pooling", ["attn_stats", "mean"])
def test_e2_padding_does_not_change_the_logit(pooling):
    cfg = M5Config(keep_layers=2, pooling=pooling, spec_augment=False)
    net = build_model(cfg, tiny=True).eval()
    rng = np.random.default_rng(0)
    clip = (0.3 * rng.standard_normal(int(0.7 * SR))).astype(np.float32)
    pad = (0.3 * rng.standard_normal(int(2.0 * SR))).astype(np.float32)
    xs1, m1 = collate([clip])
    xs2, m2 = collate([clip, pad])  # clip is now zero-padded to 2 s with a mask
    a = score_batch(net, xs1, m1)[0]
    b = score_batch(net, xs2, m2)[0]
    assert abs(float(a) - float(b)) < 1e-3


def test_e3_direction_spoof_is_one_and_score_is_sigmoid():
    y = torch.tensor([1, 0])
    hi = torch.tensor([5.0, -5.0])
    lo = torch.tensor([-5.0, 5.0])
    assert bce_with_smoothing(hi, y, 0.0) < bce_with_smoothing(lo, y, 0.0)
    assert torch.allclose(torch.sigmoid(hi), torch.tensor([0.9933, 0.0067]), atol=1e-3)


def test_e5_seeded_first_step_is_deterministic():
    losses = []
    for _ in range(2):
        torch.manual_seed(0)
        cfg = M5Config(keep_layers=2, spec_augment=False)
        net = build_model(cfg, tiny=True).train()
        losses.append(_step(net, cfg, seed=0))
    assert losses[0] == losses[1]


def test_e6_e8_save_load_roundtrip_and_hash_refusal(tmp_path):
    cfg = M5Config(keep_layers=2, spec_augment=False)
    net = build_model(cfg, tiny=True).eval()
    xs, m = _batch(3)
    ref = score_batch(net, xs.numpy(), m.numpy())
    h = save_m5(net, tmp_path / "ckpt", {"note": "test"})
    assert set(h) >= {"config_hash", "backbone_sha256", "head_sha256"}
    net2 = load_m5(tmp_path / "ckpt")
    got = score_batch(net2, xs.numpy(), m.numpy())
    assert torch.allclose(ref, got, atol=1e-5)
    assert len(net2.backbone.encoder.layers) == 2
    # tamper with the head -> refused
    p = tmp_path / "ckpt" / "head.safetensors"
    p.write_bytes(p.read_bytes()[:-1] + b"\x00")
    with pytest.raises(RuntimeError, match="sha"):
        load_m5(tmp_path / "ckpt")


def test_param_groups_llrd_and_no_decay_rules():
    cfg = M5Config(keep_layers=3, train_top=3, lr_backbone=1e-5, llrd=0.5, lr_head=5e-4)
    net = build_model(cfg, tiny=True)
    groups = param_groups(net, cfg)
    lrs = sorted({g["lr"] for g in groups})
    assert 5e-4 in lrs and 1e-5 in lrs and 5e-6 in lrs and 2.5e-6 in lrs
    head = next(g for g in groups if g["lr"] == 5e-4)
    assert head["weight_decay"] == 0.0
    ids = {id(p) for g in groups for p in g["params"]}
    assert ids == {id(p) for p in net.parameters() if p.requires_grad}  # every trainable once


def test_layer_weights_sum_to_one():
    net = build_model(M5Config(keep_layers=3), tiny=True)
    w = net.layer_weights()
    assert w.shape == (4,) and abs(float(w.sum()) - 1) < 1e-6
