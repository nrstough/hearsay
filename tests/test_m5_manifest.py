"""M5 manifest and fold discipline (spec appendix A1-A10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hearsay import m5_data
from hearsay.m5_data import (
    FOLDS,
    FOLDS_SHA256,
    build_manifest,
    check_folds_file,
    holdout_rows,
    shortcut_aucs,
    training_rows,
    validation_rows,
)  # fmt: skip


def _folds() -> pd.DataFrame:
    """A tiny fold file in the real schema: 2 inner-generator spoof groups per fold, holdout
    generators playht/wavegrad2, bona fide LJ chapters and Libri speakers spread over folds."""
    rows = []
    gens = {"0": ["grad_tts", "unit_speech"], "1": ["diffgan_tts", "openvoicev2"],
            "2": ["pro_diff", "xtts_v2"], "3": ["your_tts"], "4": ["elevenlabs"],
            "holdout": ["playht", "wavegrad2"]}  # fmt: skip
    for fold, gs in gens.items():
        for g in gs:
            for i in range(3):
                rows.append((f"/d/nsa/DiffSSD/{g}/s{i}.wav", "spoof", g, f"spk{i}", "diffssd",
                             f"gen:{g}", fold))
    for fold in ["0", "1", "2", "3", "4", "holdout"]:
        for i in range(2):
            rows.append((f"/d/lj/LJ0{fold[0]}{i}-0001.wav", "bonafide", "bonafide", "lj",
                         "ljspeech", f"lj:LJ0{fold[0]}{i}", fold))
            rows.append((f"/d/libri/{fold[0]}{i}/a.flac", "bonafide", "bonafide",
                         f"libri_{fold[0]}{i}", "librispeech", f"spk:libri_{fold[0]}{i}", fold))
    # the main chat's ASV19 extension: bona fide + anchor spoof in inner folds only
    for fold in ["0", "1"]:
        rows.append((f"/d/asv/b{fold}.flac", "bonafide", "bonafide", f"LA_00{fold}",
                     "asvspoof2019", f"asvspoof2019:LA_00{fold}", fold))
        rows.append((f"/d/asv/s{fold}.flac", "spoof", "asv19_A01", f"LA_01{fold}",
                     "asvspoof2019", "asvspoof2019:asv19_A01", fold))
    return pd.DataFrame(rows, columns=["path", "label", "generator", "speaker", "source",
                                       "group", "fold"])  # fmt: skip


def _full(folds: pd.DataFrame) -> pd.DataFrame:
    """nsa_train_full: the core rows plus extras in the SAME groups, including holdout groups."""
    core = folds[folds.source.isin(["diffssd", "ljspeech", "librispeech"])]
    utt = core.path.str.extract(r"(LJ\d{3}\d?-\d+|[^/]+)$")[0]
    base = core.assign(utt=np.where(core.source == "ljspeech",
                                    core.group.str.replace("lj:", "") + "-0001", utt))
    extra = []
    for r in base.itertuples():
        for j in range(4):
            p = r.path.replace(".wav", f"_x{j}.wav").replace(".flac", f"_x{j}.flac")
            u = r.utt if r.source != "ljspeech" else r.utt.replace("-0001", f"-000{j + 2}")
            extra.append((p, r.label, r.generator, r.speaker, u, r.source))
    cols = ["path", "label", "generator", "speaker", "utt", "source"]
    return pd.concat([base[cols], pd.DataFrame(extra, columns=cols)], ignore_index=True)


def _mlaad() -> pd.DataFrame:
    models = ["ElevenLabs-v3", "OpenVoiceV2", "tts_models/multilingual/xtts_v2", "vixTTS",
              "Gemini-3.1-Flash-TTS", "Qwen3-TTS-CustomVoice"]
    rows = [(f"/d/mlaad/{m.replace('/', '_')}/{i}.wav", "spoof", m.split("-")[0], m, "spk")
            for m in models for i in range(3)]
    return pd.DataFrame(rows, columns=["path", "label", "generator", "model_name", "speaker"])


@pytest.fixture
def manifest():
    f = _folds()
    return build_manifest(f, _full(f), _mlaad(), extra_spoof_cap=2, seed=0)


def test_a1_no_holdout_row_in_any_training_set(manifest):
    hold_paths = set(holdout_rows(manifest).path)
    hold_groups = set(holdout_rows(manifest).group)
    for k in ["0", "1", "2", "3", "4", None]:
        tr = training_rows(manifest, k)
        assert not (set(tr.path) & hold_paths)
        assert not tr.fold.eq("holdout").any()
        assert not tr.generator.isin(["playht", "wavegrad2"]).any()
        # extra rows from holdout bona fide groups (LJ chapters / Libri speakers) are gone too
        assert not tr.group.isin(hold_groups).any()


def test_a2_validation_fold_disjoint_from_training(manifest):
    for k in ["0", "1", "2", "3", "4"]:
        tr, va = training_rows(manifest, k), validation_rows(manifest, k)
        assert not (set(tr.path) & set(va.path))
        assert not tr.fold.eq(k).any()
        assert (va.train_scope == "core").all()


def test_a3_family_exclusion_and_counterfactual(manifest, monkeypatch):
    def names(k):
        return set(training_rows(manifest, k).model_name.str.lower())

    assert not any("elevenlabs" in n for n in names("4"))
    assert not any("openvoice" in n for n in names("1"))
    assert not any("xtts" in n or "vixtts" in n for n in names("2"))
    assert any("elevenlabs" in n for n in names("0"))  # other folds keep them
    assert any("elevenlabs" in n for n in names(None))  # the full model keeps them
    monkeypatch.setattr(m5_data, "FAMILY_EXCLUSIONS", {})
    assert any("elevenlabs" in n for n in names("4"))  # the guard is what removes them


def test_a4_holdout_alias_in_extra_refused():
    f = _folds()
    bad = _mlaad()
    bad.loc[0, "model_name"] = "PlayHT-2.0"
    with pytest.raises(ValueError, match="holdout family"):
        build_manifest(f, _full(f), bad)


def test_a5_extra_nsa_rows_inherit_group_fold(manifest):
    x = manifest[manifest.train_scope == "extra_nsa"]
    core = manifest[manifest.train_scope == "core"].drop_duplicates("group").set_index("group")
    assert len(x) > 0
    assert (x.fold.to_numpy() == core.loc[x.group, "fold"].to_numpy()).all()
    assert not x.fold.eq("holdout").any()
    f = _folds()
    unknown = _full(f)
    unknown.loc[len(unknown)] = ["/d/libri/99/z.flac", "bonafide", "bonafide", "libri_99",
                                 "z", "librispeech"]
    with pytest.raises(ValueError, match="absent from the fold file"):
        build_manifest(f, unknown, None)


def test_a6_duplicate_paths_dropped_and_ids_unique(manifest):
    f = _folds()
    full = _full(f)
    dup = pd.concat([full, full.iloc[[5]]], ignore_index=True)
    m = build_manifest(f, dup, _mlaad(), extra_spoof_cap=2, seed=0)
    assert m.path.is_unique and m.id.is_unique
    assert len(m) == len(manifest)


def test_a7_in_the_wild_refused():
    f = _folds()
    full = _full(f)
    full.loc[len(full)] = ["/d/in_the_wild/x.wav", "bonafide", "bonafide", "s", "u", "librispeech"]
    with pytest.raises(ValueError):
        build_manifest(f, full, None)


def test_a8_fold_file_sha_tripwire(tmp_path):
    if not FOLDS.exists():
        pytest.skip("fold file not present")
    check_folds_file(FOLDS, FOLDS_SHA256)
    p = tmp_path / "f.csv"
    p.write_text(FOLDS.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed"):
        check_folds_file(p, FOLDS_SHA256)


def test_a9_both_classes_everywhere_and_asv19_never_exported(manifest):
    for k in ["0", "1", "2", "3", "4"]:
        assert set(training_rows(manifest, k).label) == {"spoof", "bonafide"}
        assert set(validation_rows(manifest, k).label) == {"spoof", "bonafide"}
    asv = manifest[manifest.source == "asvspoof2019"]
    assert (asv.train_scope == "extra_asv19").all()
    assert not validation_rows(manifest, "0").source.eq("asvspoof2019").any()
    assert not training_rows(manifest, "0").query("source == 'asvspoof2019'").fold.eq("0").any()


def test_a10_extra_spoof_cap_is_seeded_and_per_generator():
    f = _folds()
    a = build_manifest(f, _full(f), None, extra_spoof_cap=2, seed=3)
    b = build_manifest(f, _full(f), None, extra_spoof_cap=2, seed=3)
    c = build_manifest(f, _full(f), None, extra_spoof_cap=2, seed=4)
    assert a.path.tolist() == b.path.tolist()
    assert a.path.tolist() != c.path.tolist()
    x = a[(a.train_scope == "extra_nsa") & (a.label == "spoof")]
    assert x.groupby("generator").size().max() <= 2


def test_shortcut_aucs_handles_spoof_only_scope_and_folds_direction():
    rng = np.random.default_rng(0)
    n = 200
    rows = pd.DataFrame({
        "label": ["spoof"] * 100 + ["bonafide"] * 100,
        "train_scope": ["core"] * 60 + ["extra_mlaad"] * 40 + ["core"] * 100,
        "duration": np.r_[rng.uniform(3, 6, 60), rng.uniform(8, 12, 40), rng.uniform(3, 6, 100)],
        "peak": rng.uniform(0.5, 1.0, n), "lead_s": rng.uniform(0, 0.1, n),
    })  # fmt: skip
    a = shortcut_aucs(rows)
    assert a["extra_mlaad"]["duration"] > 0.95  # long spoof-only pool vs all bona fide
    assert 0.5 <= a["core"]["duration"] < 0.7
    assert a["pooled"]["peak"] < 0.65
    rows["duration"] = -rows["duration"]  # inverted cue is just as much a cue
    assert shortcut_aucs(rows)["extra_mlaad"]["duration"] > 0.95
