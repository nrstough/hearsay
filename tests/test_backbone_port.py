"""V2-53 backbone lane: the strict key-map port, identity, and pooling.

Stages A (A1-A6), B (K5-K18), P (P1-P4) and W (W1-W4) of the run spec
`docs/specs/2026-10-11_v2-backbone.md`.

Two classes of test, per Codex plan-review finding C11 (the earlier "77 hermetic" claim was
wrong):

* **synthetic** -- a tiny AntiDeepfake-shaped checkpoint built in-process. No weights, no
  audio, no network. Covers all the name arithmetic and the numerical equivalence at toy
  scale.
* **integration** -- marked `needs_weights` + `slow` with an inner skip, run against the real
  local `weights/wav2vec2-xls-r-300m` (24 layers, hidden 1024: structurally identical to
  MMS-300M) and, when present, the downloaded AntiDeepfake checkpoints.

The local checkpoint is a `Wav2Vec2ForPreTraining` state dict in **transformers** naming with
a `wav2vec2.` prefix and *differently named* pretraining extras (`project_hid`,
`quantizer.codevectors`) than AntiDeepfake's. So the fairseq half of the map has nothing to
consume there, and the four claims it *can* support are the ones `test_k15_*` below make --
the rest of the map is first exercised against a real download in `test_w1_*`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from hearsay import antideepfake as ad
from hearsay.antideepfake import (
    HEAD,
    IDENTITY_FIELDS,
    LAYER,
    POOLING,
    PREFIX,
    STATIC,
    UNUSED,
    AntiDeepfakeDetector,
    check_identity,
    expected_mapped,
    invert_map,
    load_antideepfake,
    load_config,
    map_keys,
    max_rel_deviation,
    reference_forward,
    synth_logit,
)

REPO = ad.REPO
LOCAL_300M = REPO / "weights" / "wav2vec2-xls-r-300m"
#: The K7 gate. Measured fp32-on-CPU floor is 4.3e-08 (synthetic) and 3.6e-06 (real 24-layer),
#: and the tightest negative control clears 1e-5 by ~380x. The frozen spec said 1e-4; an
#: amendment loosened it to 1e-3 on an fp16 rationale that no longer applies, since the gate
#: runs at fp32 on CPU. 1e-5 keeps ~3x headroom over the real floor and ~100x more bite.
GATE = 1e-5

LEGACY_TO_PARAM = {
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


# --- synthetic fixtures ---------------------------------------------------------------------


def _fetch_module():
    """Import `scripts/v2_backbone_fetch.py` by path, as `tests/test_spectra.py` does.

    The first revision never imported it, so `check_target`, `file_sha256` and `hub_meta` had
    zero coverage -- including the volume check that then failed in production.
    """
    import importlib.util

    path = REPO / "scripts" / "v2_backbone_fetch.py"
    spec = importlib.util.spec_from_file_location("v2_backbone_fetch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_config(n_layers: int = 2) -> Wav2Vec2Config:
    """A small but architecturally faithful config: stable layer norm, layer conv norm."""
    return Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=n_layers, num_attention_heads=4, intermediate_size=64,
        conv_dim=(16,) * 7, conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        do_stable_layer_norm=True, feat_extract_norm="layer", conv_bias=True,
        layerdrop=0.0, apply_spec_augment=False,
    )  # fmt: skip


def fairseq_names(cfg: Wav2Vec2Config) -> list[str]:
    """Every tensor name an AntiDeepfake checkpoint of this shape would carry."""
    names = list(STATIC) + list(UNUSED) + list(HEAD)
    for i in range(cfg.num_hidden_layers):
        for sub in LAYER:
            names += [f"{PREFIX}encoder.layers.{i}.{sub}.{k}" for k in ("weight", "bias")]
    for i in range(cfg.num_feat_extract_layers):
        for which in ("0", "2.1"):
            names += [f"{PREFIX}feature_extractor.conv_layers.{i}.{which}.{k}" for k in ("weight", "bias")]
    return sorted(set(names))


def synthetic_checkpoint(cfg: Wav2Vec2Config, seed: int = 0) -> dict[str, torch.Tensor]:
    """A fairseq-named state dict with real (random) tensors of the right shapes.

    Built by instantiating a transformers model and renaming through the inverse map, so the
    shapes are exactly what the architecture demands rather than what we guess.
    """
    torch.manual_seed(seed)
    model = Wav2Vec2Model(cfg).eval().float()
    tsd = dict(model.state_dict())
    km = map_keys(fairseq_names(cfg), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    sd = {}
    # Perturb every tensor so none are accidentally identical. A freshly initialised model
    # sets every LayerNorm weight to exactly 1.0 and every bias to 0.0, which makes a swap
    # between two LayerNorm roles -- or between two biases -- a literal no-op, and the
    # negative controls in test_k7b silently vacuous. A trained checkpoint has no such
    # symmetry, so the fixture must not either.
    gen = torch.Generator().manual_seed(seed + 101)
    for src, tgt in km.mapped.items():
        t = tsd[LEGACY_TO_PARAM.get(tgt, tgt)].clone().float()
        sd[src] = t + 0.1 * torch.randn(t.shape, generator=gen)
    g = torch.Generator().manual_seed(seed + 1)
    for name in UNUSED:  # shapes are irrelevant: nothing reads them
        sd[name] = torch.zeros(1)
    sd["proj_fc.weight"] = torch.randn(2, cfg.hidden_size, generator=g) * 0.05
    sd["proj_fc.bias"] = torch.randn(2, generator=g) * 0.05
    return sd


@pytest.fixture
def cfg():
    return tiny_config()


@pytest.fixture
def ckpt(cfg):
    return synthetic_checkpoint(cfg)


@pytest.fixture
def audio():
    torch.manual_seed(7)
    x = torch.randn(1, 16000)
    return (x - x.mean()) / (x.var() + 1e-7).sqrt()


def write_ckpt(root: Path, cfg: Wav2Vec2Config, sd: dict) -> Path:
    """Materialise a synthetic checkpoint on disk: `config.json` + `model.safetensors`.

    Writing it out is the point. The first revision reimplemented the load path inside the
    test file, so every structural guard in `load_antideepfake` could be deleted with the
    whole suite still green. Going through disk means the tests exercise the shipped loader.
    """
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(cfg.to_json_string())
    save_file({k: v.contiguous() for k, v in sd.items()}, str(root / "model.safetensors"))
    return root


def port_from(cfg, sd, root: Path) -> AntiDeepfakeDetector:
    """Load a synthetic checkpoint through the real `load_antideepfake`.

    Takes `root` rather than calling `tempfile.mkdtemp()`: the latter sits outside pytest's
    `tmp_path` rotation, so ~20 calls per run left ~2 MiB of checkpoints behind permanently.
    """
    return load_antideepfake(write_ckpt(root, cfg, sd), device="cpu", dtype=torch.float32)


# === Stage B: the strict key map ============================================================


def test_k5_every_name_is_classified(cfg):
    """K5: the map is exhaustive. An unclassified name is a hard error, never a silent drop."""
    names = fairseq_names(cfg)
    km = map_keys(names, cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    assert not km.unclassified
    assert km.total == len(names)
    assert len(km.mapped) + len(km.unused) + len(km.head) == len(names)


def test_k5b_an_unknown_name_is_reported_not_swallowed(cfg):
    """K5 counterfactual: a stray tensor must land in `unclassified`."""
    km = map_keys([*fairseq_names(cfg), f"{PREFIX}encoder.mystery.weight"], cfg.num_hidden_layers)
    assert km.unclassified == frozenset({f"{PREFIX}encoder.mystery.weight"})


def test_k5c_out_of_range_layer_index_is_unclassified(cfg):
    """An off-by-one layer index must not be absorbed by a permissive regex."""
    extra = f"{PREFIX}encoder.layers.{cfg.num_hidden_layers}.fc1.weight"
    km = map_keys([*fairseq_names(cfg), extra], cfg.num_hidden_layers)
    assert extra in km.unclassified


def test_k6_every_target_parameter_is_covered(cfg, ckpt):
    """K6: no parameter may keep its random initialisation."""
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    expected = {n for n, _ in Wav2Vec2Model(cfg).named_parameters()}
    covered = {LEGACY_TO_PARAM.get(t, t) for t in km.mapped.values()}
    assert covered == expected, f"uncovered={sorted(expected - covered)} extra={sorted(covered - expected)}"


def test_k6b_a_missing_tensor_is_named(cfg, ckpt, tmp_path):
    """K6 counterfactual: dropping a tensor must raise and name it, not load silently."""
    sd = {k: v for k, v in ckpt.items() if k != f"{PREFIX}encoder.layers.0.fc1.weight"}
    km = map_keys(sd.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    expected = {n for n, _ in Wav2Vec2Model(cfg).named_parameters()}
    covered = {LEGACY_TO_PARAM.get(t, t) for t in km.mapped.values()}
    assert "encoder.layers.0.feed_forward.intermediate_dense.weight" in expected - covered


def test_k7_reference_forward_matches_the_port(tmp_path, cfg, ckpt, audio):
    """K7: the independent implementation agrees with the ported model. THE BREAK CHECK TEST.

    Gated on `max|d| / absmax(ref)` at rel 1e-3, never cosine: a deliberate q/k swap passes
    `cos > 0.99` at 0.99083.
    """
    port = port_from(cfg, ckpt, tmp_path)
    with torch.inference_mode():
        got, ref = port(audio), reference_forward(ckpt, cfg, audio)
    assert got.shape == (1, 2) and ref.shape == (1, 2)
    assert max_rel_deviation(got, ref) < GATE


@pytest.mark.parametrize(
    "label,mutate",
    [
        ("qk_swap", lambda sd, c: sd.update({
            f"{PREFIX}encoder.layers.0.self_attn.q_proj.weight": sd[f"{PREFIX}encoder.layers.0.self_attn.k_proj.weight"].clone(),
            f"{PREFIX}encoder.layers.0.self_attn.k_proj.weight": sd[f"{PREFIX}encoder.layers.0.self_attn.q_proj.weight"].clone(),
        })),
        ("kv_swap", lambda sd, c: sd.update({
            f"{PREFIX}encoder.layers.0.self_attn.k_proj.weight": sd[f"{PREFIX}encoder.layers.0.self_attn.v_proj.weight"].clone(),
            f"{PREFIX}encoder.layers.0.self_attn.v_proj.weight": sd[f"{PREFIX}encoder.layers.0.self_attn.k_proj.weight"].clone(),
        })),
        ("layernorm_roles_swapped", lambda sd, c: sd.update({
            f"{PREFIX}encoder.layers.0.self_attn_layer_norm.weight": sd[f"{PREFIX}encoder.layers.0.final_layer_norm.weight"].clone(),
            f"{PREFIX}encoder.layers.0.final_layer_norm.weight": sd[f"{PREFIX}encoder.layers.0.self_attn_layer_norm.weight"].clone(),
        })),
        ("pos_conv_randomised", lambda sd, c: sd.update({
            f"{PREFIX}encoder.pos_conv.0.weight_v": torch.randn(
                sd[f"{PREFIX}encoder.pos_conv.0.weight_v"].shape,
                generator=torch.Generator().manual_seed(4242),
            ),
        })),
    ],
)
def test_k7b_shape_compatible_mismaps_are_caught(tmp_path, cfg, ckpt, audio, label, mutate):
    """K7: swaps that keep every shape and every count intact must still be caught.

    `q/k/v/out_proj` are all `[H, H]` and 14 distinct roles share shape `(H,)`, so neither
    the tensor budget nor a shape check can see any of these.
    """
    port = port_from(cfg, ckpt, tmp_path)
    with torch.inference_mode():
        good = port(audio)
    bad_sd = {k: v.clone() for k, v in ckpt.items()}
    mutate(bad_sd, cfg)
    with torch.inference_mode():
        bad = reference_forward(bad_sd, cfg, audio)
    assert max_rel_deviation(good, bad) >= GATE, f"{label} slipped through the gate"


def test_k8_pos_conv_uses_the_legacy_weight_norm_names(cfg, ckpt):
    """K8: `weight_g`/`weight_v` map to the legacy target names, with matching shapes.

    torch 2.14 stores these as `parametrizations.weight.original0`/`original1` and installs a
    `_load_state_dict_pre_hook` that accepts the legacy spelling even at `strict=True`.
    """
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    assert km.mapped[f"{PREFIX}encoder.pos_conv.0.weight_g"] == "encoder.pos_conv_embed.conv.weight_g"
    assert km.mapped[f"{PREFIX}encoder.pos_conv.0.weight_v"] == "encoder.pos_conv_embed.conv.weight_v"
    params = dict(Wav2Vec2Model(cfg).named_parameters())
    assert ckpt[f"{PREFIX}encoder.pos_conv.0.weight_g"].shape == params[LEGACY_TO_PARAM["encoder.pos_conv_embed.conv.weight_g"]].shape
    assert ckpt[f"{PREFIX}encoder.pos_conv.0.weight_v"].shape == params[LEGACY_TO_PARAM["encoder.pos_conv_embed.conv.weight_v"]].shape


def test_k8b_the_weight_norm_axis_convention_is_right(tmp_path, cfg, ckpt):
    """K8: `g * v / ||v||` with the norm over dims != 2, checked against the live module.

    Taking the norm over dim 2 instead is an O(1) error on an O(1) tensor, so this is not a
    near-miss if it is wrong.
    """
    model = port_from(cfg, ckpt, tmp_path)
    g = ckpt[f"{PREFIX}encoder.pos_conv.0.weight_g"].double()
    v = ckpt[f"{PREFIX}encoder.pos_conv.0.weight_v"].double()
    right = (g * v / v.pow(2).sum(dim=(0, 1), keepdim=True).sqrt()).numpy()
    wrong = (g * v / v.pow(2).sum(dim=2, keepdim=True).sqrt()).numpy()
    live = model.wav2vec2.encoder.pos_conv_embed.conv.weight.detach().double().numpy()
    assert np.abs(live - right).max() < 1e-6
    assert np.abs(right - wrong).max() > 1e-2


def test_k9_a_state_dict_missing_pos_conv_raises(cfg, ckpt):
    """K9: under `strict=True` an absent tensor raises; `strict=False` would load silently."""
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    renamed = {km.mapped[k]: ckpt[k] for k in km.mapped}
    for k in ("encoder.pos_conv_embed.conv.weight_g", "encoder.pos_conv_embed.conv.weight_v"):
        renamed.pop(k)
    model = AntiDeepfakeDetector(cfg).float()
    with pytest.raises(RuntimeError, match="Missing key"):
        model.wav2vec2.load_state_dict(renamed, strict=True)
    missing = model.wav2vec2.load_state_dict(renamed, strict=False).missing_keys
    assert any("pos_conv" in m for m in missing)


def test_k10_conv_front_end_maps_conv_and_layer_norm(cfg, ckpt):
    """K10: `conv_layers.N.0` is the conv and `conv_layers.N.2.1` the LayerNorm, N bounded."""
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    for i in range(cfg.num_feat_extract_layers):
        assert km.mapped[f"{PREFIX}feature_extractor.conv_layers.{i}.0.weight"] == f"feature_extractor.conv_layers.{i}.conv.weight"
        assert km.mapped[f"{PREFIX}feature_extractor.conv_layers.{i}.2.1.weight"] == f"feature_extractor.conv_layers.{i}.layer_norm.weight"
    out_of_range = f"{PREFIX}feature_extractor.conv_layers.{cfg.num_feat_extract_layers}.0.weight"
    assert out_of_range in map_keys([out_of_range], cfg.num_hidden_layers, cfg.num_feat_extract_layers).unclassified


def test_k11_mask_emb_maps_to_masked_spec_embed(cfg, ckpt):
    """K11: the one rename that reconciles 422 targets with 421 obvious tensors."""
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    assert km.mapped[f"{PREFIX}mask_emb"] == "masked_spec_embed"


#: Transcribed from the real MMS-300M checkpoint's tensor list, independently of `UNUSED`.
#: Without this, `fairseq_names` builds its input from `UNUSED` itself, so K12 asserted
#: `UNUSED == UNUSED`: substituting a wrong-but-plausible name left 45 of 47 tests green, and
#: the only two that caught it were download-gated.
GOLDEN_UNUSED = frozenset(
    {
        "m_ssl.model.quantizer.vars",
        "m_ssl.model.quantizer.weight_proj.weight",
        "m_ssl.model.quantizer.weight_proj.bias",
        "m_ssl.model.project_q.weight",
        "m_ssl.model.project_q.bias",
        "m_ssl.model.final_proj.weight",
        "m_ssl.model.final_proj.bias",
    }
)


def test_k12_declared_unused_matches_a_transcribed_golden_list():
    """K12: the seven names pinned against a list not derived from `UNUSED`."""
    assert UNUSED == GOLDEN_UNUSED, f"drift: {sorted(UNUSED ^ GOLDEN_UNUSED)}"
    assert len(UNUSED) == 7


def test_k12b_the_seven_unused_tensors_are_declared_by_name(cfg, ckpt):
    """K12: exactly the pretraining-only seven, by name. An addition or removal must raise."""
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    assert km.unused == UNUSED and len(UNUSED) == 7
    assert all(n.startswith(PREFIX) for n in UNUSED)
    assert not (UNUSED & set(km.mapped)), "a pretraining tensor must never be mapped"


def test_k13_architecture_flags_come_from_the_shipped_config(tmp_path):
    """K13: flags are read from `config.json`, never defaulted."""
    raw = {"architectures": ["Wav2Vec2ForPreTraining"], "do_stable_layer_norm": True,
           "feat_extract_norm": "layer", "conv_bias": True, "num_hidden_layers": 4}  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = load_config(tmp_path)
    assert cfg.do_stable_layer_norm is True and cfg.feat_extract_norm == "layer" and cfg.conv_bias is True


@pytest.mark.parametrize("drop", ["do_stable_layer_norm", "feat_extract_norm", "conv_bias", "num_hidden_layers"])
def test_k13b_a_missing_flag_refuses_rather_than_defaulting(tmp_path, drop):
    """K13 counterfactual: silently defaulting an architecture flag is the failure mode."""
    raw = {"do_stable_layer_norm": True, "feat_extract_norm": "layer", "conv_bias": True, "num_hidden_layers": 4}
    raw.pop(drop)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=drop):
        load_config(tmp_path)


def test_k14_flipping_stable_layer_norm_changes_the_output(tmp_path, cfg, ckpt, audio):
    """K14: the flag is load-bearing -- every tensor can be right and the function still wrong."""
    port = port_from(cfg, ckpt, tmp_path)
    with torch.inference_mode():
        good = port(audio)
    flipped = tiny_config()
    flipped.do_stable_layer_norm = False
    other = AntiDeepfakeDetector(flipped).float()
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    other.wav2vec2.load_state_dict({km.mapped[k]: ckpt[k] for k in km.mapped}, strict=False)
    other.head.weight.data = ckpt["proj_fc.weight"].clone()
    other.head.bias.data = ckpt["proj_fc.bias"].clone()
    with torch.inference_mode():
        bad = other.eval()(audio)
    assert max_rel_deviation(good, bad) >= GATE


def test_k16_the_map_is_injective(cfg, ckpt):
    """K16: two sources claiming one target would let the later write win silently."""
    km = map_keys(ckpt.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    inv = invert_map(km.mapped)
    assert len(inv) == len(km.mapped)
    assert {inv[t] for t in inv} == set(km.mapped)


def test_k16b_a_non_injective_map_raises():
    """K16 counterfactual."""
    with pytest.raises(ValueError, match="not injective"):
        invert_map({"a": "x", "b": "x"})


def test_k17_dtype_is_what_was_requested(tmp_path, cfg, ckpt):
    """K17: a silent downcast would move every score."""
    model = port_from(cfg, ckpt, tmp_path)
    assert next(model.parameters()).dtype == torch.float32
    half = model.to(dtype=torch.float16)
    assert next(half.parameters()).dtype == torch.float16


def test_k18_layer_order_errors_are_caught(tmp_path, cfg, audio):
    """K18: all encoder layers are shape-identical, so a reordering loads with zero missing
    and zero unexpected keys. Only the numerical check can see it."""
    cfg4 = tiny_config(n_layers=4)
    sd = synthetic_checkpoint(cfg4, seed=3)
    port = port_from(cfg4, sd, tmp_path)
    with torch.inference_mode():
        good = port(audio)

    for label, perm in [("reversed", [3, 2, 1, 0]), ("off_by_one", [1, 2, 3, 0])]:
        snap = {k: v.clone() for k, v in sd.items()}
        bad_sd = {k: v.clone() for k, v in sd.items()}
        for i, src in enumerate(perm):
            for sub in LAYER:
                for kind in ("weight", "bias"):
                    bad_sd[f"{PREFIX}encoder.layers.{i}.{sub}.{kind}"] = snap[f"{PREFIX}encoder.layers.{src}.{sub}.{kind}"]
        km = map_keys(bad_sd.keys(), cfg4.num_hidden_layers, cfg4.num_feat_extract_layers)
        assert not km.unclassified and len(km.mapped) == len(map_keys(sd.keys(), cfg4.num_hidden_layers).mapped)
        with torch.inference_mode():
            bad = reference_forward(bad_sd, cfg4, audio)
        assert max_rel_deviation(good, bad) >= GATE, f"layer permutation {label} slipped through"


def test_k7c_non_finite_output_is_infinitely_deviant_never_nan(tmp_path, cfg, ckpt, audio):
    """A NaN compares False against any threshold, so it would pass a `< tol` gate silently.

    Found by the pos_conv zeroing control, which divides by a zero norm.
    """
    port = port_from(cfg, ckpt, tmp_path)
    with torch.inference_mode():
        good = port(audio)
    assert max_rel_deviation(good, torch.full_like(good, float("nan"))) == float("inf")
    assert max_rel_deviation(torch.full_like(good, float("nan")), good) == float("inf")


# === Stage A: acquisition and identity ======================================================


def identity_record(**kw) -> dict:
    base = {"model_id": "nii-yamagishilab/mms-300m-anti-deepfake", "revision": "a" * 40,
            "licence": "CC BY-NC-SA 4.0", "file_sha256": {"model.safetensors": "b" * 64},
            "realpath": "/vol/hf/x", "dtype": "float32", "n_params": 315437570,
            "download_bytes": 1269737156}  # fmt: skip
    base.update(kw)
    return base


def test_a1_revision_is_recorded_and_non_empty():
    """A1: an unpinned download means a later revision can differ silently."""
    rec = identity_record()
    assert rec["revision"] and len(rec["revision"]) == 40
    with pytest.raises(ValueError, match="revision"):
        check_identity(identity_record(revision=""), identity_record())


def test_a2_file_sha_mismatch_refuses():
    """A2: a truncated download must not be scored as if it were fine."""
    with pytest.raises(ValueError, match="file_sha256"):
        check_identity(identity_record(), identity_record(file_sha256={"model.safetensors": "c" * 64}))


def test_a3_the_shipped_volume_check_refuses_the_repo_volume(tmp_path):
    """A3: the real `check_target`, not a reimplementation of it.

    The first revision asserted a test-local prefix check, which is why the shipped code's
    "same volume, different directory" hole survived -- and `~/.cache/huggingface` is exactly
    that case.
    """
    fetch = _fetch_module()
    with pytest.raises(fetch.TargetRefused, match="same volume as the repo"):
        fetch.check_target(tmp_path, 10**6, repo=REPO)
    with pytest.raises(fetch.TargetRefused, match="same volume as the repo"):
        fetch.check_target(Path.home() / ".cache" / "huggingface", 10**6, repo=REPO)


def test_a3b_free_space_rule_is_two_times_the_download(tmp_path, monkeypatch):
    """A3: 16.4 GiB of weights against ~19 GiB free is how this project lost three builds."""
    import shutil as _sh

    fetch = _fetch_module()

    class FakeUsage:
        def __init__(self, free):
            self.free = free

    monkeypatch.setattr(_sh, "disk_usage", lambda p: FakeUsage(10**9))
    # `/dev` is devfs, a genuinely different st_dev from tmp_path, so the volume clause passes
    # and the free-space clause is the one actually under test. Asserted, not assumed.
    import os as _os

    assert _os.stat("/dev").st_dev != _os.stat(tmp_path).st_dev, "test needs two devices"
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda p: FakeUsage(10**9), raising=False)
    with pytest.raises(fetch.NotEnoughSpace, match="free"):
        fetch.check_target(tmp_path, 10**9, repo=Path("/dev"))


def test_a3c_capacity_is_checked_before_placement_and_always(tmp_path, monkeypatch):
    """A3: capacity must be enforced on every path, including the overridden one.

    The behavioural test, not a source-string one. Splitting the exception types was not the
    fix: `check_target` raised the placement error *before* computing free space, so the
    caller's override skipped capacity entirely -- a 931 GiB download against zero free space
    cleared preflight with **zero calls to `disk_usage`**. This asserts the order and the call.
    """
    import shutil as _sh

    fetch = _fetch_module()
    calls = []

    class FakeUsage:
        def __init__(self, free):
            self.free = free

    def spy(path):
        calls.append(str(path))
        return FakeUsage(0)

    monkeypatch.setattr(_sh, "disk_usage", spy)
    monkeypatch.setattr(fetch.shutil, "disk_usage", spy, raising=False)

    # same volume AND no space: capacity must win, proving it runs first
    with pytest.raises(fetch.NotEnoughSpace, match="free"):
        fetch.check_target(tmp_path, 10**9, repo=REPO)
    assert calls, "disk_usage was never called -- capacity was skipped"
    # and the device probed must be the destination, not an ancestor on another filesystem
    dest = tmp_path / "deep" / "not" / "yet" / "created"
    calls.clear()
    with pytest.raises(fetch.NotEnoughSpace):
        fetch.check_target(dest, 10**9, repo=REPO)
    assert calls == [str(dest.resolve())], f"probed {calls}, expected the destination itself"
    assert dest.exists(), "the destination must be created so its own volume is the one measured"
    assert not issubclass(fetch.NotEnoughSpace, fetch.SameVolume), (
        "a full disk must not be overridable by a placement-policy flag"
    )


def test_a6_loader_refuses_a_checkpoint_whose_bytes_do_not_match_the_record(tmp_path, cfg, ckpt):
    """A1/A6 at load time, against the bytes on disk.

    Fetch-time hashing alone let a structurally valid replacement checkpoint load under a
    stale identity record, because `check_identity` had no production caller and only compared
    dictionaries that were handed to it.
    """
    from hearsay.antideepfake import file_sha256

    root = write_ckpt(tmp_path / "bound", cfg, ckpt)

    def record(r=root):
        return {
            "model_id": "nii-yamagishilab/mms-300m-anti-deepfake",
            "revision": "a" * 40,
            "file_sha256": {
                "model.safetensors": file_sha256(r / "model.safetensors"),
                "config.json": file_sha256(r / "config.json"),
            },
        }

    good = record()
    assert load_antideepfake(root, device="cpu", dtype=torch.float32, identity=good) is not None

    # a different but structurally valid checkpoint at the same path
    write_ckpt(root, cfg, synthetic_checkpoint(cfg, seed=99))
    with pytest.raises(ValueError, match="model.safetensors sha256"):
        load_antideepfake(root, device="cpu", dtype=torch.float32, identity=good)

    # the architecture must be covered too: a shape-compatible config change alters inference
    # while the checkpoint hash still matches.
    root2 = write_ckpt(tmp_path / "cfgswap", cfg, ckpt)
    rec2 = record(root2)
    raw = json.loads((root2 / "config.json").read_text())
    raw["layerdrop"] = 0.5
    (root2 / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="config.json sha256"):
        load_antideepfake(root2, device="cpu", dtype=torch.float32, identity=rec2)

    # an incomplete record is refused rather than partially trusted
    for bad, msg in (
        ({"file_sha256": {}}, "no sha256 for model.safetensors"),
        ({"file_sha256": {"model.safetensors": "x" * 64}}, "no sha256 for config.json"),
    ):
        with pytest.raises(ValueError, match=msg):
            load_antideepfake(root, device="cpu", dtype=torch.float32, identity=bad)

    root3 = write_ckpt(tmp_path / "unpinned", cfg, ckpt)
    for field in ("model_id", "revision"):
        rec = record(root3)
        rec[field] = ""
        with pytest.raises(ValueError, match=f"no {field!r}"):
            load_antideepfake(root3, device="cpu", dtype=torch.float32, identity=rec)


def test_a6c_check_identity_rejects_two_records_that_agree_on_being_empty():
    """An earlier version compared values only, so two records agreeing a field is `""` passed
    — precisely the stale-record case this exists to catch."""
    base = identity_record()
    check_identity(base, dict(base))
    for field in IDENTITY_FIELDS:
        rec = dict(base)
        rec[field] = "" if isinstance(rec[field], str) else {}
        with pytest.raises(ValueError, match="empty"):
            check_identity(rec, dict(rec))


def test_file_sha256_matches_hashlib(tmp_path):
    """The shipped hasher, on a real file -- A2's whole mechanism."""
    import hashlib

    fetch = _fetch_module()
    f = tmp_path / "blob.bin"
    payload = b"hearsay" * 1000
    f.write_bytes(payload)
    assert fetch.file_sha256(f) == hashlib.sha256(payload).hexdigest()


def test_a4_realpath_is_recorded():
    """A4: a symlinked model dir resolving elsewhere has bitten this repo before."""
    with pytest.raises(ValueError, match="realpath"):
        check_identity(identity_record(), identity_record(realpath="/somewhere/else"))


def test_a5_every_fetched_checkpoint_records_its_licence_and_is_disclosed():
    """A5/L4: the licence is in the identity record AND the model is in the disclosure.

    The first revision set a dict key then asserted that key -- a tautology. The real
    obligation is CLAUDE.md's AI-use disclosure, which the project rubric requires for any new
    pretrained checkpoint and which is copied into Devpost.
    """
    fetch = _fetch_module()
    assert fetch.LICENCE == "CC BY-NC-SA 4.0"
    assert "licence" in IDENTITY_FIELDS
    disclosure = (REPO / "CLAUDE.md").read_text()
    records = sorted((REPO / "outputs" / "v2" / "backbone").glob("*.identity.json"))
    if not records:
        pytest.skip("no checkpoint fetched yet")
    for rec_path in records:
        rec = json.loads(rec_path.read_text())
        if rec.get("dry_run"):
            continue
        assert rec.get("licence") == "CC BY-NC-SA 4.0", f"{rec_path.name} has no licence"
        short = rec["model_id"].split("/")[-1]
        assert short in disclosure, f"{short} is on disk but absent from CLAUDE.md's disclosure"
        assert "CC BY-NC-SA" in disclosure, "the share-alike licence is not named in the disclosure"


def test_a6_identity_binding_matches_and_mismatch_raises():
    """A6: the test the user's own P2 sentence earned -- 'the right model is passed in'.

    A rank reference built from one model and scored with another is silently wrong; this is
    the `check_m5_identity` lesson from the M4/M5 lane.
    """
    rec = identity_record()
    check_identity(rec, dict(rec))
    with pytest.raises(ValueError, match="model_id"):
        check_identity(rec, identity_record(model_id="nii-yamagishilab/xls-r-2b-anti-deepfake"))


def test_a6b_an_incomplete_record_refuses(tmp_path):
    """A6: a missing field must refuse, not compare as absent-equals-absent."""
    rec = identity_record()
    del rec["revision"]
    with pytest.raises(ValueError, match="missing 'revision'"):
        check_identity(rec, identity_record())


# === Stage P: pooling and usage =============================================================


def test_p1_pooling_is_declared_and_cited():
    """P1: pooling is parameter-free, so it cannot be recovered from the checkpoint.

    It is documented: APPROACH.md:279 "adding mean pooling and a linear head with logits
    [fake, real]"; :77 and :537 "time-mean of the last layer". Run spec D6.
    """
    assert POOLING == "time_mean_last_layer"
    assert "mean pooling" in ad.__doc__ and "time-mean of the last layer" in ad.__doc__
    assert "APPROACH.md" in ad.__doc__


def test_p1b_the_declared_pooling_is_what_runs(tmp_path, cfg, ckpt, audio):
    """P1: the constant is load-bearing, not decoration."""
    port = port_from(cfg, ckpt, tmp_path)
    with torch.inference_mode():
        hidden = port.wav2vec2(audio).last_hidden_state
        expected = port.head(hidden.mean(dim=1))
        assert torch.allclose(port(audio), expected, atol=0, rtol=0)
    port.pooling = "last_frame"
    with pytest.raises(ValueError, match="unsupported pooling"):
        port(audio)


def test_p2_alternative_poolings_are_measured_not_asserted(tmp_path, cfg, ckpt, audio):
    """P2: record the margin between poolings; do not assert dominance.

    Codex C3: documented mean pooling need not beat alternatives on every sample, so turning
    that expectation into a correctness assertion could fail a correct implementation.
    """
    port = port_from(cfg, ckpt, tmp_path)
    with torch.inference_mode():
        hidden = port.wav2vec2(audio).last_hidden_state
        variants = {"time_mean_last_layer": hidden.mean(dim=1), "last_frame": hidden[:, -1, :],
                    "max_over_time": hidden.max(dim=1).values}  # fmt: skip
        logits = {k: synth_logit(port.head(v)[0]) for k, v in variants.items()}
    assert set(logits) == {"time_mean_last_layer", "last_frame", "max_over_time"}
    assert all(np.isfinite(v) for v in logits.values())
    assert logits["time_mean_last_layer"] != logits["last_frame"]


def test_p3a_CONTRACT_in_sample_ceiling_diagnostic_shape():
    """P3a **CONTRACT ONLY — not coverage of production behaviour.**

    The scorer these diagnostics belong to does not exist yet (the drive unmount blocked stage
    2), so this pins the decision *shape* against a closure defined here. It cannot detect
    missing or broken application behaviour, and the run spec records it as deferred coverage.

    P3a: a trained detector should be at ceiling on data it was post-trained on.

    Per Codex C3 this *reports and stops*, it does not declare a usage bug: corpus membership
    does not establish that these clips were training examples, nor guarantee a threshold
    after our preprocessing.
    """
    def ceiling_check(pkg03: float, eer: float, max_dcf: float = 0.05, max_eer: float = 0.02) -> dict:
        ok = pkg03 <= max_dcf and eer <= max_eer
        return {"pass": ok, "action": "continue" if ok else "stop_and_report",
                "pkg03": pkg03, "eer": eer, "declared": {"max_dcf": max_dcf, "max_eer": max_eer}}  # fmt: skip
    good = ceiling_check(0.01, 0.005)
    assert good["pass"] and good["action"] == "continue"
    bad = ceiling_check(0.40, 0.18)
    assert not bad["pass"] and bad["action"] == "stop_and_report"
    assert bad["declared"]["max_dcf"] == 0.05


def test_p3b_CONTRACT_external_corroboration_cannot_block_the_verdict():
    """P3b **CONTRACT ONLY — see `test_p3a_CONTRACT_...`.** Replaced by a test of the real
    function once the scorer exists.

    P3b: a miss sets `usage_unverified`, it never bars the verdict.

    There is no published MMS-300M In-the-Wild number and no overlapping clip set, so making
    this a gate deadlocked the lane: the verdict is its only deliverable.
    """
    def corroborate(pkg03, eer, band=(0.019, 0.098), max_eer=0.05) -> dict:
        inside = band[0] <= pkg03 <= band[1] and eer < max_eer
        return {"corroborated": inside, "usage_unverified": not inside, "blocks_verdict": False}
    hit = corroborate(0.038, 0.0144)
    assert hit["corroborated"] and not hit["usage_unverified"]
    miss = corroborate(0.42, 0.21)
    assert miss["usage_unverified"] and miss["blocks_verdict"] is False


def test_p4_CONTRACT_usage_unverified_flag_has_a_counterfactual():
    """P4: a flag nothing can set is not a flag (the M4/M5, K8/K9, K13/K14 pattern)."""
    def report(corroborated: bool) -> dict:
        return {"verdict": "column", "provisional": True, "usage_unverified": not corroborated}
    assert report(True)["usage_unverified"] is False
    assert report(False)["usage_unverified"] is True


def test_synth_logit_direction_and_arity():
    """The head is ordered [fake, real], so fake minus real rises with synthetic likelihood."""
    assert synth_logit([2.0, -1.0]) == 3.0
    assert synth_logit(np.array([-1.0, 2.0])) == -3.0
    with pytest.raises(ValueError, match="expected 2 logits"):
        synth_logit([1.0, 2.0, 3.0])


# === Stage B/W: integration against real checkpoints ========================================

needs_local = pytest.mark.skipif(
    not (LOCAL_300M / "config.json").exists(), reason="weights/wav2vec2-xls-r-300m not present"
)


@pytest.mark.needs_weights
@needs_local
def test_k15_claim1_the_422_target_budget_holds_against_a_real_file():
    """K15 (1/4): 429 = 422 base + 7 pretraining-only, measured on a real checkpoint."""
    sd = torch.load(LOCAL_300M / "pytorch_model.bin", map_location="meta", mmap=True)
    names = list(sd.keys())
    base = [n for n in names if n.startswith("wav2vec2.")]
    extras = [n for n in names if not n.startswith("wav2vec2.")]
    assert len(names) == 429 and len(base) == 422 and len(extras) == 7


@pytest.mark.needs_weights
@needs_local
def test_k15_claim2_map_targets_equal_the_models_parameter_set():
    """K15 (2/4): the map's target set is exactly what `Wav2Vec2Model` exposes."""
    cfg = load_config(LOCAL_300M)
    km = map_keys(fairseq_names(cfg), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    expected = {n for n, _ in Wav2Vec2Model(cfg).named_parameters()}
    covered = {LEGACY_TO_PARAM.get(t, t) for t in km.mapped.values()}
    assert covered == expected


@pytest.mark.needs_weights
@needs_local
def test_k15_claim3_legacy_weight_norm_names_load_at_strict_true():
    """K15 (3/4): the pos_conv remap, proven on real data before any download.

    The local checkpoint stores `weight_g`/`weight_v` and no `parametrizations.*`, so torch's
    pre-hook is exactly what makes this work.
    """
    sd = torch.load(LOCAL_300M / "pytorch_model.bin", map_location="cpu", mmap=True)
    assert any(k.endswith("pos_conv_embed.conv.weight_g") for k in sd)
    assert not any("parametrizations" in k for k in sd)
    cfg = load_config(LOCAL_300M)
    model = Wav2Vec2Model(cfg)
    stripped = {k[len("wav2vec2."):]: v for k, v in sd.items() if k.startswith("wav2vec2.")}
    model.load_state_dict(stripped, strict=True)
    conv = model.encoder.pos_conv_embed.conv
    g = dict(conv.named_parameters())["parametrizations.weight.original0"]
    assert torch.equal(g, sd["wav2vec2.encoder.pos_conv_embed.conv.weight_g"])


@pytest.mark.needs_weights
@pytest.mark.slow
@needs_local
def test_k15_claim4_equivalence_and_layer_order_on_real_24_layer_values():
    """K15 (4/4): K7 and K18 on real values at fp32 on CPU, before any download.

    This is also the test the break check mutates, so it must run from the local checkpoint
    without a download.
    """
    cfg = load_config(LOCAL_300M)
    real = Wav2Vec2Model.from_pretrained(LOCAL_300M).eval().float()
    tsd = dict(real.state_dict())
    km = map_keys(fairseq_names(cfg), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    sd = {src: tsd[LEGACY_TO_PARAM.get(tgt, tgt)].clone().float() for src, tgt in km.mapped.items()}
    torch.manual_seed(11)
    sd["proj_fc.weight"] = torch.randn(2, cfg.hidden_size) * 0.05
    sd["proj_fc.bias"] = torch.randn(2) * 0.05

    port = AntiDeepfakeDetector(cfg).float()
    port.wav2vec2.load_state_dict(tsd, strict=True)
    port.head.weight.data = sd["proj_fc.weight"].clone()
    port.head.bias.data = sd["proj_fc.bias"].clone()
    port.eval()

    x = torch.randn(1, 16000 * 2)
    x = (x - x.mean()) / (x.var() + 1e-7).sqrt()
    with torch.inference_mode():
        good, ref = port(x), reference_forward(sd, cfg, x)
    assert max_rel_deviation(good, ref) < GATE

    snap = {k: v.clone() for k, v in sd.items()}
    bad = {k: v.clone() for k, v in sd.items()}
    n = cfg.num_hidden_layers
    for i in range(n):
        for sub in LAYER:
            for kind in ("weight", "bias"):
                bad[f"{PREFIX}encoder.layers.{i}.{sub}.{kind}"] = snap[f"{PREFIX}encoder.layers.{n - 1 - i}.{sub}.{kind}"]
    with torch.inference_mode():
        reversed_out = reference_forward(bad, cfg, x)
    assert max_rel_deviation(good, reversed_out) >= GATE


def _antideepfake_records():
    """(dir, identity record) for every fetched checkpoint, so the real identity path runs."""
    out = []
    for rec_path in sorted((ad.REPO / "outputs" / "v2" / "backbone").glob("*.identity.json")):
        d = json.loads(rec_path.read_text())
        if d.get("dry_run"):
            continue
        p = Path(d["realpath"])
        if (p / "model.safetensors").exists():
            out.append((p, d))
    return out


def _antideepfake_dirs():
    """Checkpoints to exercise, located through their identity records (A6), not a glob.

    Binding to the record is the point: it is what ties a score file to the checkpoint that
    produced it. A bare path glob would happily pick up some other copy.
    """
    out = []
    for rec in sorted((ad.REPO / "outputs" / "v2" / "backbone").glob("*.identity.json")):
        d = json.loads(rec.read_text())
        if d.get("dry_run"):
            continue
        p = Path(d["realpath"])
        if (p / "model.safetensors").exists():
            out.append(p)
    out += [p for p in (ad.REPO / "weights").glob("*anti-deepfake") if (p / "model.safetensors").exists()]
    return sorted(set(out))


@pytest.mark.needs_weights
@pytest.mark.slow
def test_w1_a_real_antideepfake_checkpoint_loads_with_zero_unmapped():
    """W1: the fairseq half of the map, first exercised against a real download."""
    dirs = _antideepfake_dirs()
    if not dirs:
        pytest.skip("no AntiDeepfake checkpoint downloaded yet")
    from safetensors.torch import load_file

    for d in dirs:
        cfg = load_config(d)
        sd = load_file(str(d / "model.safetensors"))
        km = map_keys(sd.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
        assert not km.unclassified, f"{d.name}: {sorted(km.unclassified)[:5]}"
        assert km.unused == UNUSED, f"{d.name}: unused set changed"
        assert km.head == tuple(sorted(HEAD))
        expected = 422 if cfg.num_hidden_layers == 24 else 806
        assert len(km.mapped) == expected, f"{d.name}: {len(km.mapped)} mapped, expected {expected}"
        assert len(sd) == expected + 7 + 2


@pytest.mark.needs_weights
@pytest.mark.slow
def test_w4_golden_logits_are_stable_across_a_reload():
    """W4: pins the numbers against a future transformers or torch upgrade."""
    dirs = _antideepfake_dirs()
    if not dirs:
        pytest.skip("no AntiDeepfake checkpoint downloaded yet")
    recs = _antideepfake_records()
    if not recs:
        pytest.skip("no identity record for a fetched checkpoint")
    d, rec = recs[0]
    torch.manual_seed(5)
    x = torch.randn(1, 16000)
    x = (x - x.mean()) / (x.var() + 1e-7).sqrt()
    outs = []
    for _ in range(2):
        # the real identity record, so the production binding path is exercised on real bytes
        model = ad.load_antideepfake(d, device="cpu", dtype=torch.float32, identity=rec)
        with torch.inference_mode():
            outs.append(model(x).double().numpy())
        del model
    assert np.array_equal(outs[0], outs[1]), "a reload moved the logits"


# === the loader's own guards, exercised through the shipped entry point (F2) ==============


def test_expected_mapped_derives_the_budget_from_the_architecture():
    """The count is computed, not hardcoded, so every size is checked and not just the two
    we happen to download."""
    assert expected_mapped(24, 7) == 422
    assert expected_mapped(48, 7) == 806
    assert expected_mapped(2, 7) == 70


def test_load_asserts_the_tensor_budget(tmp_path, cfg, ckpt):
    """The module docstring claims the budget is "asserted on every load". It must be true on
    the production path, not only in a download-gated test."""
    root = write_ckpt(tmp_path / "ok", cfg, ckpt)
    model = load_antideepfake(root, device="cpu", dtype=torch.float32)
    assert isinstance(model, AntiDeepfakeDetector)

    extra = dict(ckpt)
    extra[f"{PREFIX}encoder.mystery.weight"] = torch.zeros(3)
    with pytest.raises(ValueError, match="unclassified"):
        load_antideepfake(write_ckpt(tmp_path / "extra", cfg, extra), device="cpu", dtype=torch.float32)


def test_load_refuses_a_missing_mapped_tensor(tmp_path, cfg, ckpt):
    """A dropped tensor must not leave a module at its random initialisation."""
    short = {k: v for k, v in ckpt.items() if k != f"{PREFIX}encoder.layers.0.fc1.weight"}
    with pytest.raises(ValueError, match="mapped tensors, expected"):
        load_antideepfake(write_ckpt(tmp_path / "short", cfg, short), device="cpu", dtype=torch.float32)


def test_load_refuses_a_removed_declared_unused_tensor(tmp_path, cfg, ckpt):
    """K12, the REMOVAL half -- and the half that had no test.

    The earlier single test dropped a name *and* added a bogus one, so it raised at the
    unclassified guard, not the declared-unused guard it was named for; a permissive
    `match="unclassified|declared-unused"` hid that. With the declared-unused guard deleted a
    6-name unused set loaded clean and the whole suite stayed green.
    """
    bad = {k: v for k, v in ckpt.items() if k != f"{PREFIX}project_q.bias"}
    with pytest.raises(ValueError, match="declared-unused set changed"):
        load_antideepfake(write_ckpt(tmp_path / "unused_removed", cfg, bad), device="cpu", dtype=torch.float32)


def test_load_refuses_an_unknown_tensor_at_the_unclassified_guard(tmp_path, cfg, ckpt):
    """K5 on the production path: the ADDITION half, asserted at its own guard."""
    bad = dict(ckpt)
    bad[f"{PREFIX}project_q.renamed"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unclassified"):
        load_antideepfake(write_ckpt(tmp_path / "unused_added", cfg, bad), device="cpu", dtype=torch.float32)


def test_load_refuses_a_missing_head_tensor(tmp_path, cfg, ckpt):
    """R2: deleting the head-tuple guard turned this into a raw KeyError from `sd[...]`."""
    bad = {k: v for k, v in ckpt.items() if k != "proj_fc.bias"}
    with pytest.raises(ValueError, match="head tensors"):
        load_antideepfake(write_ckpt(tmp_path / "nohead", cfg, bad), device="cpu", dtype=torch.float32)


def test_load_enforces_injectivity_not_merely_tests_it(tmp_path, cfg, ckpt, monkeypatch):
    """R3: `invert_map` had no caller outside tests, so a two-source collision would have been
    a silent overwrite with every count still balancing.

    The anchored index makes a natural collision unreachable, so the guard is exercised by
    introducing one into the map itself -- which is what proves the loader consults it.
    """
    import hearsay.antideepfake as mod

    collided = dict(mod.STATIC)
    collided[f"{PREFIX}mask_emb"] = "feature_projection.projection.weight"
    monkeypatch.setattr(mod, "STATIC", collided)
    with pytest.raises(ValueError, match="not injective"):
        load_antideepfake(write_ckpt(tmp_path / "collide", cfg, ckpt), device="cpu", dtype=torch.float32)


def test_the_loader_loads_strictly():
    """R2: the docstring names `strict=True` as the mechanism, so the contract gets a test.

    The count and unclassified guards now catch missing and extra tensors before
    `load_state_dict` sees them, which makes strictness redundant in practice -- but a
    documented contract with nothing behind it is exactly what the audit faulted.
    """
    import inspect

    import hearsay.antideepfake as mod

    src = inspect.getsource(mod.load_antideepfake)
    assert "load_state_dict(renamed, strict=True)" in src
    assert "strict=False" not in src


def test_load_refuses_a_wrong_shaped_head(tmp_path, cfg, ckpt):
    """F9: assigning `.data` bypassed validation, so a 3-class head loaded into an
    `nn.Linear(H, 2)` and `forward` returned (B, 3), breaking the [fake, real] contract
    before any direction gate could see it."""
    bad = dict(ckpt)
    bad["proj_fc.weight"] = torch.randn(3, cfg.hidden_size)
    bad["proj_fc.bias"] = torch.randn(3)
    with pytest.raises(ValueError, match="head shapes"):
        load_antideepfake(write_ckpt(tmp_path / "head3", cfg, bad), device="cpu", dtype=torch.float32)


def test_a_padded_layer_index_does_not_alias_a_real_layer(cfg):
    """F8: `layers.007` once mapped to the same target as `layers.7`, so the later write won
    silently with every count still balancing. The index is anchored now.

    Renamed from `test_load_refuses_...`: it asserts the classification, not a refusal by the
    loader, and the old name claimed otherwise.
    """
    for padded in (f"{PREFIX}encoder.layers.00.fc1.weight", f"{PREFIX}encoder.layers.007.fc1.weight"):
        km = map_keys([padded], cfg.num_hidden_layers, cfg.num_feat_extract_layers)
        assert padded in km.unclassified, f"{padded} must not alias a real layer"


def test_load_refuses_bad_architecture_flags(tmp_path, cfg, ckpt):
    """F7: the flags were checked for presence, not value. A config with
    `do_stable_layer_norm: false` loaded clean and computed a different function."""
    root = write_ckpt(tmp_path / "flags", cfg, ckpt)
    raw = json.loads((root / "config.json").read_text())
    raw["do_stable_layer_norm"] = False
    (root / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="do_stable_layer_norm"):
        load_antideepfake(root, device="cpu", dtype=torch.float32)

    raw["do_stable_layer_norm"] = True
    raw["feat_extract_norm"] = "group"
    (root / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="feat_extract_norm"):
        load_antideepfake(root, device="cpu", dtype=torch.float32)


def test_load_config_names_the_file_on_malformed_json(tmp_path):
    """F20: a bare JSONDecodeError with no path, and `null` raised an opaque TypeError."""
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(tmp_path)
    (tmp_path / "config.json").write_text("null")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_config(tmp_path)


def test_k17_dtype_mismatch_is_refused_not_silently_honoured(tmp_path, cfg, ckpt):
    """F12: the old test asserted `.float()` then `nn.Module.to` -- it could never fail, and
    the on-disk dtype was never compared to anything."""
    root = write_ckpt(tmp_path / "fp32", cfg, ckpt)
    assert next(load_antideepfake(root, device="cpu", dtype=torch.float32).parameters()).dtype == torch.float32
    with pytest.raises(ValueError, match="checkpoint is torch.float32"):
        load_antideepfake(root, device="cpu", dtype=torch.float16)

    half = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt.items()}
    root16 = write_ckpt(tmp_path / "fp16", cfg, half)
    assert next(load_antideepfake(root16, device="cpu", dtype=torch.float16).parameters()).dtype == torch.float16
    with pytest.raises(ValueError, match="checkpoint is torch.float16"):
        load_antideepfake(root16, device="cpu", dtype=torch.float32)

    # R9: an fp16 base with an fp32 head once loaded clean as fp32, silently upcasting the base,
    # because the guard sampled a single tensor.
    mixed = dict(half)
    mixed["proj_fc.weight"] = mixed["proj_fc.weight"].float()
    mixed["proj_fc.bias"] = mixed["proj_fc.bias"].float()
    with pytest.raises(ValueError, match="mixed tensor dtypes"):
        load_antideepfake(write_ckpt(tmp_path / "mixed", cfg, mixed), device="cpu", dtype=torch.float32)


def test_reference_forward_refuses_a_missing_conv_bias(cfg, ckpt, audio):
    """F13: `sd.get` returned a finite, wrong answer from the one function whose entire
    purpose is to be the independent check."""
    bad = {k: v for k, v in ckpt.items() if k != f"{PREFIX}feature_extractor.conv_layers.0.0.bias"}
    with pytest.raises(KeyError):
        reference_forward(bad, cfg, audio)


def test_max_rel_deviation_refuses_mismatched_shapes(cfg, ckpt, audio):
    """F14: broadcasting turned a batch mismatch into a plausible number -- (1,2) vs (8,2)
    returned 0.5 -- and K7/K18 both gate on that number."""
    with pytest.raises(ValueError, match="shape mismatch"):
        max_rel_deviation(torch.zeros(1, 2), torch.ones(8, 2))
    with pytest.raises(ValueError, match="all zeros"):
        max_rel_deviation(torch.zeros(2), torch.zeros(2))


def test_synth_logit_refuses_non_finite():
    """F20: the same NaN-swallowing class already fixed in `max_rel_deviation`."""
    with pytest.raises(ValueError, match="non-finite"):
        synth_logit([float("nan"), 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        synth_logit([float("inf"), 1.0])


def test_mps_reservation_blocks_a_second_claimant(tmp_path):
    """F17: the first revision's "lock" was a text file no code consulted."""
    from hearsay.antideepfake import mps_reservation

    lock = tmp_path / "MPS.lock"
    with mps_reservation(lock, scope="first"):
        assert lock.exists()
        with pytest.raises(TimeoutError, match="refusing to share"), mps_reservation(
            lock, scope="second", wait_s=0.1
        ):
            pass
    assert not lock.exists(), "the lock must be released on exit"


def test_mps_reservation_reclaims_a_stale_lock_whose_owner_is_gone(tmp_path):
    """A killed job must not wedge the device forever."""
    from hearsay.antideepfake import mps_reservation

    lock = tmp_path / "MPS.lock"
    lock.write_text("pid=999999999\ntoken=deadbeef\n")  # a pid that cannot exist
    # reclaimed on the spot, with no wait: a dead owner's lock is never worth waiting for.
    # An earlier version gated liveness behind `age > stale_s`, so a killed job's lock blocked
    # the device for the full 12-hour default -- which it then did, to a real run.
    with mps_reservation(lock, scope="x", stale_s=43200.0, wait_s=0.0):
        assert lock.exists()

    # an unparseable lock file has no discoverable owner, so age is the only signal left
    lock2 = tmp_path / "other.lock"
    lock2.write_text("garbage\n")
    with mps_reservation(lock2, scope="y", stale_s=0.0, wait_s=0.0):
        assert lock2.exists()


def test_mps_reservation_will_not_steal_a_lock_from_a_live_owner(tmp_path):
    """R5: it reclaimed purely on age, so a second claimant took the device out from under a
    live holder still inside its `with` block -- reinstating the concurrent-2B condition."""
    import os

    from hearsay.antideepfake import mps_reservation

    lock = tmp_path / "MPS.lock"
    lock.write_text(f"pid={os.getpid()}\ntoken=other\n")  # alive by construction
    # stale_s=0 on purpose: a LIVE owner must never be reclaimed from, at any age. An earlier
    # revision made liveness and age a single `or`, so this case started stealing the lock.
    with pytest.raises(TimeoutError, match="refusing to share"), mps_reservation(
        lock, scope="thief", wait_s=0.1, stale_s=0.0
    ):
        pass
    assert "token=other" in lock.read_text(), "a live owner's lock must survive"


def test_mps_reservation_releases_only_its_own_lock(tmp_path):
    """R6: release unlinked whatever was there, so after a reclaim our exit deleted the other
    job's live lock -- two lanes ping-ponging the device."""
    from hearsay.antideepfake import mps_reservation

    lock = tmp_path / "MPS.lock"
    with mps_reservation(lock, scope="first"):
        lock.write_text("pid=1\ntoken=someone_else\n")  # someone else now holds it
    assert lock.exists(), "we must not unlink a lock we no longer own"
    assert "someone_else" in lock.read_text()


@pytest.mark.needs_weights
@pytest.mark.needs_data
@pytest.mark.slow
def test_w2_real_audio_scores_finite_at_every_length():
    """W2: real In-the-Wild audio through the real checkpoint, at the lengths that actually
    occur. Recorded as part of the Stage 1 gate but absent from the first revision."""
    dirs = _antideepfake_dirs()
    if not dirs:
        pytest.skip("no AntiDeepfake checkpoint downloaded yet")
    man = REPO / "outputs" / "manifests" / "itw_stress.csv"
    if not man.exists():
        pytest.skip("itw_stress manifest absent")
    import csv

    from hearsay.antideepfake import mps_reservation
    from hearsay.audio import load_audio
    from hearsay.embed import prepare_segment

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg absent; hearsay.audio.load_audio needs it")
    rows = list(csv.DictReader(man.open()))[:4]
    paths = [REPO / r["path"] for r in rows]
    if not all(p.exists() for p in paths):
        pytest.skip("In-the-Wild audio not reachable")

    # A legitimate scoring run holding the device is not a test failure. The reservation
    # working correctly must not turn into a red suite, so a held lock skips.
    try:
        ctx = mps_reservation(
            REPO / "outputs" / "v2" / "backbone" / "MPS.lock", scope="test_w2", wait_s=5.0
        )
        ctx.__enter__()
    except TimeoutError as e:
        pytest.skip(f"MPS device reserved by another job: {e}")
    try:
        model = load_antideepfake(dirs[0], device="cpu", dtype=torch.float32)
        for p in paths:
            x = prepare_segment(load_audio(p))
            assert x.size > 0
            t = torch.from_numpy(x)[None]
            t = (t - t.mean()) / (t.var() + 1e-7).sqrt()
            with torch.inference_mode():
                logits = model(t)
            assert logits.shape == (1, 2)
            assert np.isfinite(logits.numpy()).all()
            assert np.isfinite(synth_logit(logits[0]))
    finally:
        ctx.__exit__(None, None, None)


def test_i6_clips_below_the_receptive_field_are_refused_by_name(tmp_path, cfg, ckpt):
    """C4/I6: the declared short-input policy, at the boundary rather than "whatever the first
    four manifest rows happen to be".

    Below one encoder frame both paths used to raise an opaque
    `RuntimeError: Kernel size can't be greater than actual input size`.
    """
    from hearsay.antideepfake import receptive_field

    need = receptive_field(cfg.conv_kernel, cfg.conv_stride)
    assert need == 400, f"shipped kernels/strides give 400 samples, got {need}"
    port = port_from(cfg, ckpt, tmp_path)

    for n in (1, need - 1):
        x = torch.zeros(1, n)
        with pytest.raises(ValueError, match="below the 400-sample"):
            port(x)
        with pytest.raises(ValueError, match="below the 400-sample"):
            reference_forward(ckpt, cfg, x)

    for n in (need, need + 1, need * 4):
        x = torch.randn(1, n)
        x = (x - x.mean()) / (x.var() + 1e-7).sqrt()
        with torch.inference_mode():
            out = port(x)
            ref = reference_forward(ckpt, cfg, x)
        assert out.shape == (1, 2) and np.isfinite(out.numpy()).all()
        assert max_rel_deviation(out, ref) < GATE


GOLDEN = Path(__file__).parent / "fixtures" / "backbone_golden.json"


def test_w4b_golden_logits_are_persisted_not_just_self_consistent(tmp_path, cfg, ckpt):
    """C4: comparing two reloads under the same installed libraries cannot see library drift.

    The values are written to a committed fixture on first run and compared thereafter, so a
    `transformers` or `torch` upgrade that moves the numbers is visible instead of silent.
    """
    port = port_from(cfg, ckpt, tmp_path)
    torch.manual_seed(1234)
    x = torch.randn(1, 16000)
    x = (x - x.mean()) / (x.var() + 1e-7).sqrt()
    with torch.inference_mode():
        got = port(x).double().numpy().ravel().tolist()

    if not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"tiny_cfg_seed0_logits": got}, indent=1) + "\n")
        pytest.skip(f"golden written to {GOLDEN.name}; commit it and the comparison runs from now on")

    want = json.loads(GOLDEN.read_text())["tiny_cfg_seed0_logits"]
    assert np.allclose(got, want, rtol=1e-6, atol=1e-8), (
        f"logits drifted from the committed golden: {got} vs {want}. If this is an intended "
        "library upgrade, update the fixture and say so in the run spec."
    )
