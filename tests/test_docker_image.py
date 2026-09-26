"""K, the Docker image: hermetic checks on the build artifacts and the docker/ scripts (run spec
docs/specs/2026-09-26_k-docker-image.md, tests B1-B10, P2-P4, R-static). Nothing here needs the
image, weights, data or the network; the image itself is exercised by docker/smoke.sh."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hearsay import SR

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = (REPO / "Dockerfile").read_text()
DOCKERIGNORE = (REPO / ".dockerignore").read_text().splitlines()
ENTRYPOINT = (REPO / "docker" / "entrypoint.sh").read_text()
BUILD = (REPO / "docker" / "build.sh").read_text()
SMOKE = (REPO / "docker" / "smoke.sh").read_text()
SCRIPTS = [REPO / "docker" / n for n in ("entrypoint.sh", "build.sh", "smoke.sh")]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lock_version(pkg: str) -> str:
    m = re.search(rf'^name = "{pkg}"\nversion = "([^"]+)"', (REPO / "uv.lock").read_text(), re.MULTILINE)
    assert m, f"{pkg} not in uv.lock"
    return m.group(1)


# --- B1-B6: Dockerfile ----------------------------------------------------------------------


def test_b1_base_image_and_ffmpeg_layer():
    assert re.search(r"^FROM python:3\.12-slim-bookworm@sha256:[0-9a-f]{64}$", DOCKERFILE, re.MULTILINE)
    apt = re.search(r"^RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg"
                    r" && rm -rf /var/lib/apt/lists/\*$", DOCKERFILE, re.MULTILINE)
    assert apt, "ffmpeg must be installed and the apt lists removed in one layer"


def test_b2_offline_env_baked_in():
    env = re.search(r"^ENV .*?(?=^\w)", DOCKERFILE, re.MULTILINE | re.DOTALL).group(0)
    for var in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
                "PYTHONPATH=/app/src", "NUMBA_CACHE_DIR=/tmp/numba"):
        assert var in env, var


def test_b3_locked_deps_cpu_torch_no_cuda_no_lightgbm():
    run = re.search(r"^RUN uv export.*?rm -rf /root/\.cache/uv$", DOCKERFILE, re.MULTILINE | re.DOTALL).group(0)
    assert "--frozen" in run and "--no-dev" in run and "--no-hashes" in run and "--no-emit-project" in run
    for pkg in ("torch", "torchaudio", "lightgbm"):
        assert f"--no-emit-package {pkg}" in run, pkg
    assert "grep -vE '^(nvidia-|cuda-|triton)'" in run
    assert "uv pip install --python /opt/venv/bin/python --no-deps -r /app/requirements-cpu.txt" in run, \
        "--no-deps is load-bearing: without it speechbrain pulls CUDA torch from PyPI"
    torch_v, ta_v = _lock_version("torch"), _lock_version("torchaudio")
    assert f"--index-url https://download.pytorch.org/whl/cpu torch=={torch_v} torchaudio=={ta_v}" in run
    assert "uv pip check" in run
    assert "README.md" in DOCKERFILE.split("RUN uv export")[0], "the editable install needs README.md"


def test_b4_context_excludes_and_never_copies_private_trees():
    for name in (".git", ".venv", "data", "outputs", "submissions", "models", "weights/wavlm-*",
                 "tests", "docs", ".env", "*.wav"):
        assert name in DOCKERIGNORE, name
    copies = [ln.split()[1:-1] for ln in DOCKERFILE.splitlines() if ln.startswith("COPY ")]
    sources = [s for srcs in copies for s in srcs if not s.startswith("--")]
    for bad in ("data", "outputs", "submissions", "wavlm", ".venv", ".git", "tests"):
        assert not any(s == bad or s.startswith(bad + "/") or f"/{bad}" in s for s in sources), (bad, sources)
    assert sources, "no COPY sources parsed"


def test_b5_context_keeps_what_the_image_copies():
    kept = ("weights/wav2vec2-xls-r-300m", "weights/Spectra-AASIST", "weights/spkrec-ecapa-voxceleb",
            "src", "scripts", "docker", "pyproject.toml", "uv.lock", "README.md")
    for k in kept:
        assert k not in DOCKERIGNORE and not any(
            ln and not ln.startswith("#") and k.startswith(ln.rstrip("/") + "/") for ln in DOCKERIGNORE), k
    assert "weights" not in DOCKERIGNORE, "no wholesale weights exclusion (a ! re-include is not relied on)"


def test_b6_copy_destinations_match_the_loaders():
    import hearsay.detectors._learned as learned
    from hearsay import embed, pipeline

    for name in ("wav2vec2-xls-r-300m", "Spectra-AASIST", "spkrec-ecapa-voxceleb"):
        assert f"COPY weights/{name} ./weights/{name}" in DOCKERFILE, name
        assert (embed.REPO / "weights" / name).parts[-2:] == ("weights", name)
    assert "COPY docker/build/models ./models" in DOCKERFILE
    assert learned.MODELS.name == "models" and pipeline.HC_DIR.parts[-2:] == ("models", "hc_selected")
    assert pipeline.PROBE_DIR.parent.name == "models" and pipeline.CONSTANTS_PATH.parts[-3:] == \
        ("models", "fusion_v0", "constants.json")
    # weights before code: a code edit must not re-copy the weight layers
    assert DOCKERFILE.index("COPY weights/wav2vec2-xls-r-300m") < DOCKERFILE.index("COPY src ./src")
    assert DOCKERFILE.index("COPY src ./src") < DOCKERFILE.index("assets.py freeze")


# --- B7-B9: docker/ scripts -----------------------------------------------------------------


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_b8_scripts_parse_and_are_strict(path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "set -euo pipefail" in path.read_text()


def test_b8_entrypoint_contract_matches_the_runner():
    assert ('exec python /app/scripts/run_pipeline.py --in /data --out /out --threads "$OMP_NUM_THREADS" '
            '--require-offline "$@"') in ENTRYPOINT
    assert "/sys/fs/cgroup/cpu.max" in ENTRYPOINT and "OMP_NUM_THREADS" in ENTRYPOINT
    assert '[ "$n" -gt 6 ] && n=6' in ENTRYPOINT, "threads are capped at six, as the README says"
    assert "assets.py verify" in ENTRYPOINT and "HEARSAY_TEMPLATE" in ENTRYPOINT
    text = (REPO / "scripts" / "run_pipeline.py").read_text()
    for flag in ('"--in"', '"--out"', '"--threads"', '"--template"', '"--team"', '"--rule"',
                 '"--require-offline"'):
        assert flag in text, flag
    for env in ("HEARSAY_TEAM", "HEARSAY_TEMPLATE", "HEARSAY_RULE"):
        assert env in text, env


def _fake_tree(root: Path) -> None:
    (root / "models" / "m1_real").mkdir(parents=True)
    (root / "models" / "m1_real" / "probe.joblib").write_bytes(b"p")
    (root / "models" / "hc_lgbm_x").mkdir()
    (root / "models" / "hc_lgbm_x" / "model.joblib").write_bytes(b"h")
    (root / "models" / "cmp_lgbm_x").mkdir()
    (root / "models" / "cmp_lgbm_x" / "model.joblib").write_bytes(b"c")
    for v in ("fusion_v0", "fusion_v1"):
        (root / "models" / v).mkdir()
        (root / "models" / v / "constants.json").write_text("{}")
    os.symlink("hc_lgbm_x", root / "models" / "hc_selected")
    os.symlink("cmp_lgbm_x", root / "models" / "cmp_selected")


def test_b7_build_resolves_symlinked_bundles(tmp_path):
    _fake_tree(tmp_path)
    env = {**os.environ, "REPO_ROOT": str(tmp_path), "PROBE_DIR": "models/m1_real", "MIN_FREE_GB": "0"}
    r = subprocess.run(["bash", str(REPO / "docker" / "build.sh"), "--print-args"], env=env,
                       capture_output=True, text=True, check=False, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert f"HC_REAL={tmp_path.resolve() / 'models' / 'hc_lgbm_x'}" in r.stdout, r.stdout
    assert f"CMP_REAL={tmp_path.resolve() / 'models' / 'cmp_lgbm_x'}" in r.stdout
    assert "hc=hc_lgbm_x" in r.stdout and "probe=m1_real" in r.stdout
    assert "fusion=fusion_v0,fusion_v1" in r.stdout and "FUSION_DIRS=models/fusion_v0 models/fusion_v1" in r.stdout
    # dangling symlink: loud failure, no build
    (tmp_path / "models" / "hc_selected").unlink()
    os.symlink("gone", tmp_path / "models" / "hc_selected")
    r = subprocess.run(["bash", str(REPO / "docker" / "build.sh"), "--print-args"], env=env,
                       capture_output=True, text=True, check=False, cwd=tmp_path)
    assert r.returncode != 0 and "hc_selected" in r.stderr
    # the disk guard refuses a build (even a dry run) below MIN_FREE_GB
    (tmp_path / "models" / "hc_selected").unlink()
    os.symlink("hc_lgbm_x", tmp_path / "models" / "hc_selected")
    env["MIN_FREE_GB"] = "100000"
    r = subprocess.run(["bash", str(REPO / "docker" / "build.sh"), "--print-args"], env=env,
                       capture_output=True, text=True, check=False, cwd=tmp_path)
    assert r.returncode == 1 and "GB free on the drive hosting" in r.stderr


def test_b9_no_lightgbm_on_the_image_path():
    for p in (REPO / "docker" / "assets.py", REPO / "docker" / "parity.py",
              REPO / "scripts" / "run_pipeline.py", REPO / "src" / "hearsay" / "pipeline.py"):
        for ln in p.read_text().splitlines():
            assert not ln.strip().startswith(("import lightgbm", "from lightgbm")), p
    _load(REPO / "docker" / "assets.py", "assets_mod")
    _load(REPO / "docker" / "parity.py", "parity_mod")
    assert "lightgbm" not in sys.modules


# --- B10: shipped-asset manifest ------------------------------------------------------------


def test_b10_assets_freeze_verify_and_tamper(tmp_path):
    assets = _load(REPO / "docker" / "assets.py", "assets_mod")
    (tmp_path / "models" / "a").mkdir(parents=True)
    (tmp_path / "models" / "a" / "f.bin").write_bytes(b"x" * 10)
    (tmp_path / "models" / "a" / "__pycache__").mkdir()
    (tmp_path / "models" / "a" / "__pycache__" / "z.pyc").write_bytes(b"ignored")
    (tmp_path / "weights" / "w").mkdir(parents=True)
    (tmp_path / "weights" / "w" / "big.bin").write_bytes(b"y" * 100)
    m = assets.freeze(tmp_path, ["models/a", "weights/w"])
    assert {f["path"] for f in m["files"]} == {"models/a/f.bin", "weights/w/big.bin"}
    assert assets.verify(tmp_path, m) == []
    assert assets.verify(tmp_path, m, full=True) == []
    # size-only fast path skips hashing large files; --full catches a same-size tamper
    (tmp_path / "weights" / "w" / "big.bin").write_bytes(b"z" * 100)
    assert assets.verify(tmp_path, m, hash_over_mb=0.00005) == []
    assert assets.verify(tmp_path, m, full=True) == ["sha256 mismatch: weights/w/big.bin"]
    (tmp_path / "models" / "a" / "f.bin").write_bytes(b"x" * 11)
    both = ["size mismatch: models/a/f.bin (11 != 10)", "sha256 mismatch: weights/w/big.bin"]
    assert assets.verify(tmp_path, m) == both  # big.bin is under 100 MB, so it is hashed by default
    assert assets.verify(tmp_path, m, full=True) == both
    assert assets.verify(tmp_path, m, hash_over_mb=0.00005) == both[:1]  # size-only fast path
    (tmp_path / "models" / "a" / "extra.txt").write_text("!")
    assert "extra: models/a/extra.txt" in assets.verify(tmp_path, m)
    # the manifest itself may live inside a frozen dir (the 06:46 smoke failure): never "extra"
    inside = tmp_path / "models" / "a" / "assets.json"
    inside.write_text(json.dumps(m))
    assert "extra: models/a/assets.json" in assets.verify(tmp_path, m)  # counterfactual
    assert "extra: models/a/assets.json" not in assets.verify(tmp_path, m, manifest_path=inside)
    assert assets.main(["verify", "--root", str(tmp_path), "--manifest", str(inside)]) == 1  # extra.txt only
    inside.unlink()
    (tmp_path / "models" / "a" / "f.bin").unlink()
    assert "missing: models/a/f.bin" in assets.verify(tmp_path, m)
    # CLI exit codes
    out = tmp_path / "m.json"
    rc = assets.main(["freeze", "--root", str(tmp_path), "--dirs", "weights/w", "--out", str(out)])
    assert rc == 0 and json.loads(out.read_text())["files"][0]["path"] == "weights/w/big.bin"
    assert assets.main(["verify", "--root", str(tmp_path), "--manifest", str(out)]) == 0
    (tmp_path / "weights" / "w" / "big.bin").unlink()
    assert assets.main(["verify", "--root", str(tmp_path), "--manifest", str(out)]) == 1
    with pytest.raises(SystemExit):
        assets.freeze(tmp_path, ["models/nope"])


# --- P2-P4: parity tool and smoke script ----------------------------------------------------


def test_p2_pcm_hash_detects_a_changed_sample(tmp_path):
    parity = _load(REPO / "docker" / "parity.py", "parity_mod")
    rng = np.random.default_rng(0)
    x = (0.3 * rng.standard_normal(SR)).astype(np.float32)
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir(), b.mkdir()
    for d in (a, b):
        sf.write(d / "one.wav", x, SR, subtype="PCM_16")
        sf.write(d / "two.wav", x[::-1], SR, subtype="PCM_16")
    y = x.copy()
    y[100] += 0.01
    sf.write(b / "two.wav", y[::-1], SR, subtype="PCM_16")  # one sample differs
    (tmp_path / "t.tsv").write_text("filename\tcm-score\ntwo.wav\t0.5\none.wav\t0.5\n")
    ha = parity.pcm_hash(a, tmp_path / "t.tsv", None, tmp_path / "a.json")
    hb = parity.pcm_hash(b, tmp_path / "t.tsv", 2, tmp_path / "b.json")
    assert list(ha) == ["two.wav", "one.wav"] and ha["one.wav"] == hb["one.wav"]
    assert ha["one.wav"]["n_samples"] == SR
    bad = parity.pcm_diff(tmp_path / "a.json", tmp_path / "b.json")
    assert bad == [f"two.wav: pcm differs ({SR} vs {SR} samples)"]
    assert parity.main(["pcm-diff", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 1
    assert parity.main(["pcm-diff", str(tmp_path / "a.json"), str(tmp_path / "a.json")]) == 0
    # two failed decodes (empty hash, zero samples) must never count as a match
    (a / "junk.wav").write_bytes(b"not audio" * 8)
    (b / "junk.wav").write_bytes(b"not audio" * 8)
    (tmp_path / "t2.tsv").write_text("filename\tcm-score\njunk.wav\t0.5\none.wav\t0.5\n")
    parity.pcm_hash(a, tmp_path / "t2.tsv", None, tmp_path / "a2.json")
    parity.pcm_hash(b, tmp_path / "t2.tsv", None, tmp_path / "b2.json")
    bad = parity.pcm_diff(tmp_path / "a2.json", tmp_path / "b2.json")
    assert bad == ["junk.wav: decode failed (both); a failed decode never counts as a match"]
    # one-sided failure is labelled with its side
    sf.write(b / "junk.wav", x[:SR // 2], SR, subtype="PCM_16")
    parity.pcm_hash(b, tmp_path / "t2.tsv", None, tmp_path / "b3.json")
    bad = parity.pcm_diff(tmp_path / "a2.json", tmp_path / "b3.json")
    assert bad == ["junk.wav: decode failed (A); a failed decode never counts as a match"]


def test_p3_smoke_script_runs_offline_and_checks_order():
    assert "--network none" in SMOKE and ":/data:ro" in SMOKE
    assert "--entrypoint python" in SMOKE and "assets.py verify" in SMOKE and "--full" in SMOKE
    assert "-e TRANSFORMERS_OFFLINE=0" in SMOKE and "refused without the offline variables" in SMOKE, \
        "the negative offline check (D2) is part of the smoke test"
    neg = SMOKE.index("-e TRANSFORMERS_OFFLINE=0")
    assert SMOKE.index("c_tone.flac") < neg < SMOKE.index("for i in 1 2"), \
        "the negative check runs on the real input, before the positive runs"
    assert "-e HEARSAY_TEAM=smoke" in SMOKE[neg - 200:neg + 200] and 'ls "$ROOT/outneg"/*.tsv' in SMOKE
    assert "c_tone.flac b_noise.mp3 a_sine.wav" in SMOKE, "reversed template order is asserted"
    assert "max |diff|" in SMOKE and "1e-6" in SMOKE, "repeat runs are compared with a tolerance"
    assert "outputs/docker/smoke" in SMOKE and "/tmp" not in SMOKE.replace("/tmpl/", ""), \
        "Colima bind-mounts only paths under $HOME"


def test_p4_build_script_targets_amd64_and_records_provenance():
    assert "--platform linux/amd64" in BUILD and "--load" in BUILD
    assert "--build-arg BUILD_INFO=" in BUILD and "--build-arg GIT_SHA=" in BUILD
    assert "cp -RL" in BUILD and "models/fusion_*" in BUILD and "cmp_selected" in BUILD
    assert "ARG GIT_SHA" in DOCKERFILE and "ENV HEARSAY_GIT_SHA=$GIT_SHA" in DOCKERFILE
    # speaker drift offline (the 06:53 smoke failure): the detector now loads ECAPA from weights/
    # only (commit 1cc4fbe); the image ships the dir untouched and the smoke test proves it runs
    assert "pretrained_path" not in DOCKERFILE and "speaker_drift OK" in SMOKE
    assert "COPY weights/spkrec-ecapa-voxceleb ./weights/spkrec-ecapa-voxceleb" in DOCKERFILE
    assert (REPO / "docker" / ".gitignore").read_text().strip() == "build/"
    # the system disk hosts the VM's sparse disk: a free-space guard before, pruning after
    assert "MIN_FREE_GB" in BUILD and 'df -Pk "$VM_DIR"' in BUILD
    assert "docker image prune -f" in BUILD and "docker builder prune" not in BUILD, \
        "old tags and dangling images go; the layer cache stays for the next build"
