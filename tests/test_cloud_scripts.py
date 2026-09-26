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
    assert "xlsr_hf_revision=hf_rev" in b and "xlsr_hf_weight_sha256=hf_sha" in b
    assert "FATAL: XLS-R config sha" in t
    assert "xlsr_hf_weight_sha256" in t and "sha-verified" in t  # downloaded weights are checked
    assert "HEARSAY_R2_PREFIX}weights" not in t  # no unverified R2 weights path


def test_g2_launcher_destroys_a_box_whose_chain_never_starts(tmp_path):
    """Run launch.sh against stubbed `uvx vastai` (creates a 'running' box) and a stubbed `ssh`
    that succeeds for the network check but fails to start the chain: the launcher must exit
    non-zero, destroy the instance, and never write PROVISIONED."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = tmp_path / "vast.log"
    (stubs / "uvx").write_text(f"""#!/bin/bash
echo "$@" >> {log}
case "$*" in
  *'show user'*) echo '{{"credit": 30.0}}';;
  *'create instance'*) echo '{{"new_contract": 4242}}';;
  *'show instances'*) echo '[{{"id": 4242, "actual_status": "running", "ssh_host": "h", "ssh_port": "1", "dph_total": 0.6}}]';;
  *'destroy instance'*) echo destroyed;;
esac
""")
    (stubs / "ssh").write_text(r"""#!/bin/bash
# the network check passes; rclone setup passes; the chain start never reports CHAIN-UP
for a in "$@"; do case "$a" in *NET-OK*) echo NET-OK; exit 0;; *RCLONE-OK*) cat >/dev/null; echo RCLONE-OK; exit 0;; *tar\ xzf*) cat >/dev/null; exit 0;; esac; done
exit 1
""")
    (stubs / "rclone").write_text("#!/bin/bash\nexit 0\n")
    for f in stubs.iterdir():
        f.chmod(0o755)
    creds = tmp_path / "creds.txt"
    creds.write_text("account_id=a\naccess_key_id=k\nsecret_access_key=s\n")
    home = tmp_path / "home"
    (home / ".config" / "vastai").mkdir(parents=True)
    (home / ".config" / "vastai" / "vast_api_key").write_text("stub-key")
    (home / ".config" / "cloudflare-r2-pa-source.txt").write_text(creds.read_text())
    import os

    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home)}
    r = subprocess.run(["bash", str(CLOUD / "launch.sh"), "tjob", "fold=0", "0.5", "111"],
                       capture_output=True, text=True, cwd=REPO, env=env, check=False)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "PROVISIONED" not in r.stdout
    assert "destroy instance 4242" in log.read_text()
    assert not (home / ".hearsay_vast" / "tjob" / "PROVISIONED").exists()
    # the instance id was persisted at creation, before anything could fail
    assert (home / ".hearsay_vast" / "tjob" / "CID").read_text().strip() == "4242"


def test_g2_launcher_retries_destroy_and_leaves_an_orphan_marker(tmp_path):
    """Same failure, but the first destroy call does not take: the launcher retries, and if the
    box is still listed it leaves CID + ORPHAN for the reaper instead of forgetting it."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = tmp_path / "vast.log"
    (stubs / "uvx").write_text(f"""#!/bin/bash
echo "$@" >> {log}
n=$(grep -c 'destroy instance' {log} 2>/dev/null || echo 0)
case "$*" in
  *'show user'*) echo '{{"credit": 30.0}}';;
  *'create instance'*) echo '{{"new_contract": 4343}}';;
  *'show instances'*) if [ "$n" -ge 2 ]; then echo '[]'; else echo '[{{"id": 4343, "actual_status": "running", "ssh_host": "h", "ssh_port": "1", "dph_total": 0.6}}]'; fi;;
  *'destroy instance'*) echo ok;;
esac
""")
    (stubs / "ssh").write_text("#!/bin/bash\nfor a in \"$@\"; do case \"$a\" in *NET-OK*) echo NET-OK; exit 0;; esac; done\nexit 1\n")
    (stubs / "rclone").write_text("#!/bin/bash\nexit 0\n")
    for f in stubs.iterdir():
        f.chmod(0o755)
    home = tmp_path / "home"
    (home / ".config" / "vastai").mkdir(parents=True)
    (home / ".config" / "vastai" / "vast_api_key").write_text("stub")
    (home / ".config" / "cloudflare-r2-pa-source.txt").write_text("account_id=a\naccess_key_id=k\nsecret_access_key=s\n")
    import os

    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home), "DESTROY_TRIES": "2",
           "DESTROY_WAIT": "0"}
    r = subprocess.run(["bash", str(CLOUD / "launch.sh"), "tjob2", "fold=0", "0.5", "222"],
                       capture_output=True, text=True, cwd=REPO, env=env, check=False)
    assert r.returncode != 0
    assert log.read_text().count("destroy instance 4343") == 2  # retried until absent
    assert (home / ".hearsay_vast" / "tjob2" / "CID").read_text().strip() == "4343"
    assert not (home / ".hearsay_vast" / "tjob2" / "ORPHAN").exists()  # gone on the 2nd try


def test_g2_reaper_retries_until_the_instance_is_gone(tmp_path):
    """Stubbed vastai: the first destroy silently fails (instance still listed), the second
    works. DESTROYED must appear only after the instance is absent."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    state = tmp_path / "state"
    (state / "job").mkdir(parents=True)
    (state / "job" / "CID").write_text("77")
    calls = tmp_path / "calls"
    (stubs / "uvx").write_text(f"""#!/bin/bash
echo "$@" >> {calls}
n=$(grep -c 'destroy instance' {calls} 2>/dev/null || echo 0)
case "$*" in
  *'show instances'*) if [ "$n" -ge 2 ]; then echo '[]'; else echo '[{{"id": 77}}]'; fi;;
  *'destroy instance'*) echo ok;;
esac
""")
    (stubs / "rclone").write_text("#!/bin/bash\necho DONE\n")  # STATUS says DONE
    for f in stubs.iterdir():
        f.chmod(0o755)
    home = tmp_path / "home"
    (home / ".config" / "vastai").mkdir(parents=True)
    (home / ".config" / "vastai" / "vast_api_key").write_text("stub")
    import os

    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home),
           "POLL": "1", "DEADLINE": str(int(__import__("time").time()) + 3600)}
    # run the reaper's loop body three times by replacing its infinite loop with a bounded one
    src = (CLOUD / "reaper.sh").read_text().replace('STATE="$HOME/.hearsay_vast"', f'STATE="{state}"')
    src = src.replace('. "$HERE/r2_guard.sh"', f'. "{CLOUD}/r2_guard.sh"')  # the copy lives in tmp
    src = src.replace('REPO="$(cd "$HERE/../.." && pwd)"', f'REPO="{REPO}"')  # real .venv python
    src = src.replace('LEDGER="$REPO/docs/reports/cloud-expense-ledger.md"', f'LEDGER="{tmp_path}/ledger.md"')
    src = src.replace("while :; do", "for _i in 1 2 3; do", 1).replace("sleep 5\n", "sleep 0\n")
    script = tmp_path / "reaper_bounded.sh"
    script.write_text(src)
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=REPO, env=env,
                       check=False, timeout=60)
    assert "not confirmed; retrying" in r.stdout, r.stdout + r.stderr
    assert (state / "job" / "DESTROYED").exists()
    assert calls.read_text().count("destroy instance 77") == 2


def _stub_home(tmp_path):
    home = tmp_path / "home"
    (home / ".config" / "vastai").mkdir(parents=True)
    (home / ".config" / "vastai" / "vast_api_key").write_text("stub")
    (home / ".config" / "cloudflare-r2-pa-source.txt").write_text("account_id=a\naccess_key_id=k\nsecret_access_key=s\n")
    return home


def test_g2_launcher_refuses_to_reuse_a_job_whose_instance_is_still_listed(tmp_path):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = tmp_path / "vast.log"
    (stubs / "uvx").write_text(f"""#!/bin/bash
echo "$@" >> {log}
case "$*" in
  *'show user'*) echo '{{"credit": 30.0}}';;
  *'show instances'*) echo '[{{"id": 9000, "actual_status": "running", "dph_total": 0.6}}]';;
  *'create instance'*) echo '{{"new_contract": 9001}}';;
esac
""")
    (stubs / "uvx").chmod(0o755)
    home = _stub_home(tmp_path)
    (home / ".hearsay_vast" / "oldjob").mkdir(parents=True)
    (home / ".hearsay_vast" / "oldjob" / "CID").write_text("9000")
    (home / ".hearsay_vast" / "oldjob" / "ORPHAN").write_text("1")
    import os

    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home)}
    r = subprocess.run(["bash", str(CLOUD / "launch.sh"), "oldjob", "fold=0", "0.5", "1"],
                       capture_output=True, text=True, cwd=REPO, env=env, check=False)
    assert r.returncode == 4 and "REFUSED" in r.stderr
    assert "create instance" not in log.read_text()
    # the record survives for the reaper
    assert (home / ".hearsay_vast" / "oldjob" / "CID").read_text() == "9000"
    assert (home / ".hearsay_vast" / "oldjob" / "ORPHAN").exists()


def test_g2_reaper_destroys_a_held_job_past_the_deadline_and_expires_stale_holds(tmp_path):
    import time

    stubs = tmp_path / "bin"
    stubs.mkdir()
    state = tmp_path / "state"
    (state / "held").mkdir(parents=True)
    (state / "held" / "CID").write_text("55")
    (state / "held" / "HOLD").write_text(str(int(time.time())))  # a fresh hold
    calls = tmp_path / "calls"
    (stubs / "uvx").write_text(f"""#!/bin/bash
echo "$@" >> {calls}
case "$*" in
  *'show instances'*) if grep -q 'destroy instance' {calls} 2>/dev/null; then echo '[]'; else echo '[{{"id": 55}}]'; fi;;
  *'destroy instance'*) echo ok;;
esac
""")
    (stubs / "rclone").write_text("#!/bin/bash\necho RUNNING\n")
    for f in stubs.iterdir():
        f.chmod(0o755)
    home = _stub_home(tmp_path)
    import os

    base_env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home), "POLL": "1"}
    src = (CLOUD / "reaper.sh").read_text().replace('STATE="$HOME/.hearsay_vast"', f'STATE="{state}"')
    src = src.replace('. "$HERE/r2_guard.sh"', f'. "{CLOUD}/r2_guard.sh"')
    src = src.replace('REPO="$(cd "$HERE/../.." && pwd)"', f'REPO="{REPO}"')
    src = src.replace('LEDGER="$REPO/docs/reports/cloud-expense-ledger.md"', f'LEDGER="{tmp_path}/ledger.md"')
    src = src.replace("while :; do", "for _i in 1 2; do", 1).replace("sleep 5\n", "sleep 0\n")
    script = tmp_path / "reaper_bounded.sh"
    script.write_text(src)
    # held + deadline passed -> destroyed anyway
    env = {**base_env, "DEADLINE": str(int(time.time()) - 10)}
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=REPO, env=env,
                       check=False, timeout=60)
    assert "deadline" in r.stdout and "destroy instance 55" in calls.read_text(), r.stdout + r.stderr
    assert (state / "held" / "DESTROYED").exists()
    # a stale hold (older than HOLD_MAX_MIN) is expired and the job is watched again
    calls.write_text("")
    (state / "held2").mkdir()
    (state / "held2" / "CID").write_text("56")
    (state / "held2" / "HOLD").write_text(str(int(time.time()) - 3600))
    env = {**base_env, "DEADLINE": str(int(time.time()) + 3600), "HOLD_MAX_MIN": "15"}
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=REPO, env=env,
                       check=False, timeout=60)
    assert "stale hold expired" in r.stdout and not (state / "held2" / "HOLD").exists()


def test_g2_launcher_keeps_the_record_when_the_instance_query_fails(tmp_path):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "uvx").write_text("#!/bin/bash\ncase \"$*\" in *'show instances'*) echo 'not json'; exit 1;; *'show user'*) echo '{\"credit\": 30}';; esac\n")
    (stubs / "uvx").chmod(0o755)
    home = _stub_home(tmp_path)
    (home / ".hearsay_vast" / "j").mkdir(parents=True)
    (home / ".hearsay_vast" / "j" / "CID").write_text("31")
    import os

    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home)}
    r = subprocess.run(["bash", str(CLOUD / "launch.sh"), "j", "fold=0", "0.5", "1"],
                       capture_output=True, text=True, cwd=REPO, env=env, check=False)
    assert r.returncode == 4 and "not confirmed" in r.stderr
    assert (home / ".hearsay_vast" / "j" / "CID").read_text() == "31"


def test_g2_reaper_fresh_hold_defers_a_stale_fail_status(tmp_path):
    import time

    stubs = tmp_path / "bin"
    stubs.mkdir()
    state = tmp_path / "state"
    (state / "sw").mkdir(parents=True)
    (state / "sw" / "CID").write_text("61")
    (state / "sw" / "HOLD").write_text(str(int(time.time())))
    calls = tmp_path / "calls"
    (stubs / "uvx").write_text(f"#!/bin/bash\necho \"$@\" >> {calls}\ncase \"$*\" in *'show instances'*) echo '[{{\"id\": 61}}]';; *) echo ok;; esac\n")
    (stubs / "rclone").write_text("#!/bin/bash\necho 'FAIL full_frozen'\n")  # stale status from before the swap
    for f in stubs.iterdir():
        f.chmod(0o755)
    home = _stub_home(tmp_path)
    import os

    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home), "POLL": "1",
           "DEADLINE": str(int(time.time()) + 3600)}
    src = (CLOUD / "reaper.sh").read_text().replace('STATE="$HOME/.hearsay_vast"', f'STATE="{state}"')
    src = src.replace('. "$HERE/r2_guard.sh"', f'. "{CLOUD}/r2_guard.sh"')
    src = src.replace('REPO="$(cd "$HERE/../.." && pwd)"', f'REPO="{REPO}"')
    src = src.replace('LEDGER="$REPO/docs/reports/cloud-expense-ledger.md"', f'LEDGER="{tmp_path}/ledger.md"')
    src = src.replace("while :; do", "for _i in 1 2; do", 1)
    script = tmp_path / "reaper_bounded.sh"
    script.write_text(src)
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=REPO, env=env,
                       check=False, timeout=60)
    assert "held" in r.stdout and "deferred" in r.stdout, r.stdout + r.stderr
    assert "destroy instance" not in (calls.read_text() if calls.exists() else "")
    assert not (state / "sw" / "DESTROYED").exists()
