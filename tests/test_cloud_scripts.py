"""Static and mocked checks on the M5 cloud scripts (spec appendix G1-G6, F8)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLOUD = REPO / "scripts" / "cloud"
SHELL = sorted(CLOUD.glob("*.sh"))
M5_PY = sorted(REPO.glob("scripts/m5_*.py")) + [CLOUD / "box_codecs.py",
                                                REPO / "src/hearsay/m5_data.py",
                                                REPO / "src/hearsay/m5_bundle.py",
                                                REPO / "src/hearsay/m5_model.py",
                                                REPO / "src/hearsay/augment.py"]


def test_scripts_exist_and_parse():
    assert {p.name for p in SHELL} >= {"launch.sh", "reaper.sh", "box_setup.sh", "box_chain.sh",
                                        "r2_guard.sh", "teardown_check.sh", "pull_runs.sh"}
    for p in SHELL:
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, f"{p.name}: {r.stderr}"


def test_g1_every_r2_path_is_under_the_hearsay_prefix():
    pat = re.compile(r"r2:[A-Za-z0-9_./${}-]+")
    for p in SHELL + M5_PY:
        for line in p.read_text().splitlines():
            if line.strip().startswith("#"):
                continue
            for m in pat.findall(line):
                assert m.startswith("r2:pa-source/hearsay/") or m in ("r2:pa-source/hearsay", "r2:*"), (
                    p.name, line.strip())


def test_g1_guard_function_refuses_other_prefixes():
    good = subprocess.run(["bash", "-c", f". {CLOUD}/r2_guard.sh; r2_path r2:pa-source/hearsay/x"],
                          capture_output=True, text=True, check=False)
    bad = subprocess.run(["bash", "-c", f". {CLOUD}/r2_guard.sh; r2_path r2:pa-source/bundle/x"],
                         capture_output=True, text=True, check=False)
    other = subprocess.run(["bash", "-c", f". {CLOUD}/r2_guard.sh; r2_copy /tmp/x r2:pa-source/training/"],
                           capture_output=True, text=True, check=False)
    assert good.returncode == 0 and good.stdout.strip() == "r2:pa-source/hearsay/x"
    assert bad.returncode != 0 and "REFUSED" in bad.stderr
    assert other.returncode != 0 and "REFUSED" in other.stderr


def test_g2_launcher_trap_and_reaper_branches():
    launch = (CLOUD / "launch.sh").read_text()
    assert "trap cleanup EXIT" in launch and "kill_ \"$CID\"" in launch
    reaper = (CLOUD / "reaper.sh").read_text()
    for needle in ("DONE)", "FAIL*)", "DEADLINE", "stalled", "kill_"):
        assert needle in reaper
    chain = (CLOUD / "box_chain.sh").read_text()
    assert 'status "DONE"' in chain and 'fail()' in chain and "vastai" not in chain


def test_g3_budget_guard_arithmetic_with_a_stub_vastai(tmp_path):
    """Run only the guard block of launch.sh against a stubbed `uvx vastai`."""
    src = (CLOUD / "launch.sh").read_text()
    guard = src[src.index("# --- budget guard"): src.index("ist() {")]
    stub = tmp_path / "uvx"
    stub.write_text("#!/bin/bash\ncase \"$*\" in\n  *'show user'*) echo '{\"credit\": 5.0}';;\n"
                    "  *'show instances'*) echo '[{\"id\": 1, \"dph_total\": 0.7}]';;\nesac\n")
    stub.chmod(0o755)
    script = (f'export PATH="{tmp_path}:$PATH"; REPO="{REPO}"; VAST="uvx vastai"; '
              'py() { "$REPO/.venv/bin/python" -c "$@"; }; EST_HOURS=$1\n' + guard + "\necho PASSED")
    ok = subprocess.run(["bash", "-c", script, "_", "1"], capture_output=True, text=True, check=False)
    assert "PASSED" in ok.stdout, ok.stdout + ok.stderr  # 5.0 - 1.4 >= 0.9 + 1.5
    no = subprocess.run(["bash", "-c", script, "_", "4"], capture_output=True, text=True, check=False)
    assert no.returncode == 3 and "REFUSED" in no.stderr  # 5.0 - 1.4 < 3.6 + 1.5


def test_g5_key_never_printed_or_shipped():
    for p in SHELL:
        t = p.read_text()
        assert "set -x" not in t, p.name
        assert "cat $KEY" not in t and 'echo "$KEY"' not in t and "echo $KEY" not in t, p.name
    # the box never receives the vast key: the chain has no VAST_API_KEY and the launcher's
    # ssh command that starts the chain passes only JOB/JOBS/DEADLINE
    launch = (CLOUD / "launch.sh").read_text()
    start = [ln for ln in launch.splitlines() if "box_chain.sh" in ln and "setsid" in ln]
    assert start and all("VAST_API_KEY" not in ln and "$KEY" not in ln for ln in start)
    assert "VAST_API_KEY" not in (CLOUD / "box_chain.sh").read_text()
    assert "VAST_API_KEY" not in (CLOUD / "box_setup.sh").read_text()


def test_f8_m5_code_never_touches_other_lanes():
    for p in M5_PY:
        t = p.read_text()
        assert "submissions/" not in t, p.name
        assert ".tsv" not in t.lower() or "never" in t.lower(), p.name
        assert not re.search(r"nsa_folds(_plus_asv19)?\.csv[\"']?\s*,\s*[\"']w", t), p.name
        assert "write_submission" not in t, p.name


@pytest.mark.parametrize("name", ["box_setup.sh", "box_chain.sh"])
def test_box_scripts_source_the_guard(name):
    assert "r2_guard.sh" in (CLOUD / name).read_text()


def test_g6_box_setup_pins_the_hf_revision_and_checks_the_config_sha():
    t = (CLOUD / "box_setup.sh").read_text()
    assert "revision=rev" in t and 'get("xlsr_hf_revision")' in t
    assert 'assert rev, "bundle_meta.json has no xlsr_hf_revision' in t  # refuses unpinned
    b = (REPO / "scripts" / "m5_build_bundle.py").read_text()
    assert "xlsr_hf_revision=hf_rev" in b  # the key is produced by the bundle build itself
    assert "FATAL: XLS-R config sha" in t and "SHA256SUMS" in t
