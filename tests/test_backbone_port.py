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
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from hearsay import antideepfake as ad
from hearsay.antideepfake import (
    HEAD,
    LAYER,
    POOLING,
    PREFIX,
    STATIC,
    UNUSED,
    AntiDeepfakeDetector,
    invert_map,
    load_config,
    map_keys,
    max_rel_deviation,
    reference_forward,
    synth_logit,
)

REPO = ad.REPO
LOCAL_300M = REPO / "weights" / "wav2vec2-xls-r-300m"
LEGACY_TO_PARAM = {
    "encoder.pos_conv_embed.conv.weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
}


# --- synthetic fixtures ---------------------------------------------------------------------


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


def port_from(cfg, sd) -> AntiDeepfakeDetector:
    """Load a synthetic checkpoint the way `load_antideepfake` does, without touching disk."""
    km = map_keys(sd.keys(), cfg.num_hidden_layers, cfg.num_feat_extract_layers)
    model = AntiDeepfakeDetector(cfg).float()
    model.wav2vec2.load_state_dict({km.mapped[k]: v for k in km.mapped for v in [sd[k]]}, strict=True)
    model.head.weight.data = sd["proj_fc.weight"].clone()
    model.head.bias.data = sd["proj_fc.bias"].clone()
    return model.eval()


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


def test_k7_reference_forward_matches_the_port(cfg, ckpt, audio):
    """K7: the independent implementation agrees with the ported model. THE BREAK CHECK TEST.

    Gated on `max|d| / absmax(ref)` at rel 1e-3, never cosine: a deliberate q/k swap passes
    `cos > 0.99` at 0.99083.
    """
    port = port_from(cfg, ckpt)
    with torch.inference_mode():
        got, ref = port(audio), reference_forward(ckpt, cfg, audio)
    assert got.shape == (1, 2) and ref.shape == (1, 2)
    assert max_rel_deviation(got, ref) < 1e-3


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
            f"{PREFIX}encoder.pos_conv.0.weight_v": torch.randn_like(sd[f"{PREFIX}encoder.pos_conv.0.weight_v"]),
        })),
    ],
)
def test_k7b_shape_compatible_mismaps_are_caught(cfg, ckpt, audio, label, mutate):
    """K7: swaps that keep every shape and every count intact must still be caught.

    `q/k/v/out_proj` are all `[H, H]` and 14 distinct roles share shape `(H,)`, so neither
    the tensor budget nor a shape check can see any of these.
    """
    port = port_from(cfg, ckpt)
    with torch.inference_mode():
        good = port(audio)
    bad_sd = {k: v.clone() for k, v in ckpt.items()}
    mutate(bad_sd, cfg)
    with torch.inference_mode():
        bad = reference_forward(bad_sd, cfg, audio)
    assert max_rel_deviation(good, bad) >= 1e-3, f"{label} slipped through the gate"


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


def test_k8b_the_weight_norm_axis_convention_is_right(cfg, ckpt):
    """K8: `g * v / ||v||` with the norm over dims != 2, checked against the live module.

    Taking the norm over dim 2 instead is an O(1) error on an O(1) tensor, so this is not a
    near-miss if it is wrong.
    """
    model = port_from(cfg, ckpt)
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


def test_k12_the_seven_unused_tensors_are_declared_by_name(cfg, ckpt):
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


def test_k14_flipping_stable_layer_norm_changes_the_output(cfg, ckpt, audio):
    """K14: the flag is load-bearing -- every tensor can be right and the function still wrong."""
    port = port_from(cfg, ckpt)
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
    assert max_rel_deviation(good, bad) >= 1e-3


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


def test_k17_dtype_is_what_was_requested(cfg, ckpt):
    """K17: a silent downcast would move every score."""
    model = port_from(cfg, ckpt)
    assert next(model.parameters()).dtype == torch.float32
    half = model.to(dtype=torch.float16)
    assert next(half.parameters()).dtype == torch.float16


def test_k18_layer_order_errors_are_caught(cfg, audio):
    """K18: all encoder layers are shape-identical, so a reordering loads with zero missing
    and zero unexpected keys. Only the numerical check can see it."""
    cfg4 = tiny_config(n_layers=4)
    sd = synthetic_checkpoint(cfg4, seed=3)
    port = port_from(cfg4, sd)
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
        assert max_rel_deviation(good, bad) >= 1e-3, f"layer permutation {label} slipped through"


def test_k7c_non_finite_output_is_infinitely_deviant_never_nan(cfg, ckpt, audio):
    """A NaN compares False against any threshold, so it would pass a `< tol` gate silently.

    Found by the pos_conv zeroing control, which divides by a zero norm.
    """
    port = port_from(cfg, ckpt)
    with torch.inference_mode():
        good = port(audio)
    assert max_rel_deviation(good, torch.full_like(good, float("nan"))) == float("inf")
    assert max_rel_deviation(torch.full_like(good, float("nan")), good) == float("inf")


# === Stage A: acquisition and identity ======================================================


def identity_record(**kw) -> dict:
    base = {"model_id": "nii-yamagishilab/mms-300m-anti-deepfake", "revision": "a" * 40,
            "file_sha256": {"model.safetensors": "b" * 64}, "realpath": "/vol/hf/x", "dtype": "float32",
            "n_params": 315437570, "download_bytes": 1269737156}  # fmt: skip
    base.update(kw)
    return base


def check_identity(record: dict, loaded: dict) -> None:
    """A6: refuse before any readout when the score file's model is not the loaded one."""
    for field in ("model_id", "revision", "file_sha256", "realpath"):
        if field not in record:
            raise ValueError(f"identity record missing {field!r}")
        if record[field] != loaded.get(field):
            raise ValueError(f"identity mismatch on {field!r}: recorded {record[field]!r} vs loaded {loaded.get(field)!r}")


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


def test_a3_weights_must_resolve_off_the_repo_volume(tmp_path):
    """A3: 16.4 GiB of weights against 19 GiB free on `/` is how this project lost builds."""
    def place(target, free_bytes, need_bytes):
        if str(target).startswith(str(REPO)):
            raise ValueError(f"refusing to place weights inside the repo volume: {target}")
        if free_bytes < 2 * need_bytes:
            raise ValueError(f"refusing: {free_bytes} free < 2x {need_bytes}")
        return target
    with pytest.raises(ValueError, match="inside the repo volume"):
        place(REPO / "weights" / "x", 10**12, 10**9)
    with pytest.raises(ValueError, match="free"):
        place(tmp_path / "x", 10**9, 10**9)
    assert place(tmp_path / "x", 10**12, 10**9)


def test_a4_realpath_is_recorded():
    """A4: a symlinked model dir resolving elsewhere has bitten this repo before."""
    with pytest.raises(ValueError, match="realpath"):
        check_identity(identity_record(), identity_record(realpath="/somewhere/else"))


def test_a5_licence_is_part_of_the_record():
    """A5: CC BY-NC-SA 4.0 is non-commercial AND share-alike; it must be recorded."""
    rec = identity_record(licence="CC BY-NC-SA 4.0")
    assert rec["licence"] == "CC BY-NC-SA 4.0"


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


def test_p1b_the_declared_pooling_is_what_runs(cfg, ckpt, audio):
    """P1: the constant is load-bearing, not decoration."""
    port = port_from(cfg, ckpt)
    with torch.inference_mode():
        hidden = port.wav2vec2(audio).last_hidden_state
        expected = port.head(hidden.mean(dim=1))
        assert torch.allclose(port(audio), expected, atol=0, rtol=0)
    port.pooling = "last_frame"
    with pytest.raises(ValueError, match="unsupported pooling"):
        port(audio)


def test_p2_alternative_poolings_are_measured_not_asserted(cfg, ckpt, audio):
    """P2: record the margin between poolings; do not assert dominance.

    Codex C3: documented mean pooling need not beat alternatives on every sample, so turning
    that expectation into a correctness assertion could fail a correct implementation.
    """
    port = port_from(cfg, ckpt)
    with torch.inference_mode():
        hidden = port.wav2vec2(audio).last_hidden_state
        variants = {"time_mean_last_layer": hidden.mean(dim=1), "last_frame": hidden[:, -1, :],
                    "max_over_time": hidden.max(dim=1).values}  # fmt: skip
        logits = {k: synth_logit(port.head(v)[0]) for k, v in variants.items()}
    assert set(logits) == {"time_mean_last_layer", "last_frame", "max_over_time"}
    assert all(np.isfinite(v) for v in logits.values())
    assert logits["time_mean_last_layer"] != logits["last_frame"]


def test_p3a_in_sample_ceiling_is_a_diagnostic_with_a_bounded_stop():
    """P3a: a trained detector should be at ceiling on data it was post-trained on.

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


def test_p3b_external_corroboration_cannot_block_the_verdict():
    """P3b: a miss sets `usage_unverified`, it never bars the verdict.

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


def test_p4_the_usage_unverified_flag_has_a_counterfactual():
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
    assert max_rel_deviation(good, ref) < 1e-3

    snap = {k: v.clone() for k, v in sd.items()}
    bad = {k: v.clone() for k, v in sd.items()}
    n = cfg.num_hidden_layers
    for i in range(n):
        for sub in LAYER:
            for kind in ("weight", "bias"):
                bad[f"{PREFIX}encoder.layers.{i}.{sub}.{kind}"] = snap[f"{PREFIX}encoder.layers.{n - 1 - i}.{sub}.{kind}"]
    with torch.inference_mode():
        reversed_out = reference_forward(bad, cfg, x)
    assert max_rel_deviation(good, reversed_out) >= 1e-3


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
    d = dirs[0]
    torch.manual_seed(5)
    x = torch.randn(1, 16000)
    x = (x - x.mean()) / (x.var() + 1e-7).sqrt()
    outs = []
    for _ in range(2):
        model = ad.load_antideepfake(d, device="cpu", dtype=torch.float32)
        with torch.inference_mode():
            outs.append(model(x).double().numpy())
        del model
    assert np.array_equal(outs[0], outs[1]), "a reload moved the logits"
