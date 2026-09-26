"""Doc-code consistency tripwires (run spec docs/specs/2026-09-25_software-only-rescope.md, A1-A8).

Living docs must not drift back to the dropped hardware scope or the CSV wording, must state
the same metric constants as `hearsay.metrics`, and the plan must cover every scored
technique. Historical docs are frozen by design (A7).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from hearsay import metrics

REPO = Path(__file__).resolve().parents[1]
CORE = [REPO / "CLAUDE.md", REPO / "README.md", REPO / "docs" / "plan.md"]
SKILLS = sorted((REPO / ".claude" / "skills").glob("*/SKILL.md"))
RUBRICS = sorted((REPO / ".claude").glob("codex-*.md"))
LIVING = CORE + SKILLS + RUBRICS
BASE_SHA = "e1a1e88"
BANNER_PREFIX = "> **Historical (pre-rescope, Fri Sep 25):**"
CUT = "(cut)"
MAX_CUT_LINES = 6

HARDWARE = re.compile(
    r"listening[- ]head|pan[- ]tilt|\bservos?\b|mic[- ]arrays?|\b4[- ]mic\b|head[- ]driver"
    r"|direction[- ]finding|\blasers?\b|grandma|live[- ]demo|replay[- ]capture|firmware"
    r"|\bpucks?\b|\bgauges?\b",
    re.IGNORECASE,
)
LED = re.compile(r"\bLEDs?\b")  # case-sensitive: "led" inside "labeled" must not match
CSV_WORD = re.compile(r"\bCSVs?\b")  # case-sensitive; ".csv" paths are allowlisted per line
CSV_ALLOW = ("log.csv", "csv_path", ".csv", CUT)


def section(text: str, heading: str) -> str:
    """Body of `## heading` up to the next `## ` heading (### subsections stay inside)."""
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert m, f"missing section '## {heading}'"
    return m.group(1)


def hardware_hits(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if HARDWARE.search(ln) or LED.search(ln)]


# --- guards against a vacuous suite --------------------------------------------------------


def test_living_doc_set_is_complete():
    assert all(p.exists() for p in CORE)
    assert len(SKILLS) >= 9, SKILLS
    assert {p.name for p in RUBRICS} == {"codex-review-prompt.md", "codex-audit-prompt.md"}


def test_hardware_regex_self_test():
    assert hardware_hits("the pan-tilt servos aim the laser")
    assert hardware_hits("a 4-mic array and a live demo")
    assert hardware_hits("status LED turns red")
    clean = "header, labeled, microsoft, issue tracker, demonstrates, headless, called, deliverable"
    assert not hardware_hits(clean)


# --- A1 no hardware scope in living docs ---------------------------------------------------


@pytest.mark.parametrize("path", LIVING, ids=lambda p: str(p.relative_to(REPO)))
def test_a1_no_hardware_terms(path):
    hits = hardware_hits(path.read_text())
    exempt = [ln for ln in hits if CUT in ln]
    assert len(exempt) <= MAX_CUT_LINES, f"too many (cut) exemptions: {exempt}"
    offending = [ln for ln in hits if CUT not in ln]
    assert not offending, offending


# --- A2 plan covers the scored techniques and deliverables ---------------------------------


def test_a2_plan_names_every_technique_and_deliverable():
    plan = (REPO / "docs" / "plan.md").read_text()
    tech = section(plan, "Forensic techniques")
    for pat in [r"metadata|container", r"spectral", r"prosod", r"\bENF\b", r"compression",
                r"speaker[- ]embedding", r"anti-spoof", r"splice", r"orchestrat"]:  # fmt: skip
        assert re.search(pat, tech, re.IGNORECASE), pat
    for pat in [r"Docker", r"README", r"TSV", r"MinDCF", r"8 AM"]:
        assert re.search(pat, plan, re.IGNORECASE), pat


# --- A3 metric constants agree between code and docs ---------------------------------------

C_FA_DOC = re.compile(r"C_FA\s*=\s*4(\.0)?\b")
PI_DOC = re.compile(r"π_synth\s*=\s*0\.3\b")
PI_CONFLICT = re.compile(r"(π|pi)_?synth\s*=\s*0\.(?!3\b)\d+", re.IGNORECASE)


def test_a3_regex_self_test():
    assert C_FA_DOC.search("`C_FA = 4`") and C_FA_DOC.search("C_FA=4.0")
    assert PI_DOC.search("π_synth = 0.3,") and not PI_DOC.search("π_synth = 0.35")
    assert PI_CONFLICT.search("pi_synth = 0.5") and PI_CONFLICT.search("π_synth=0.25")
    assert not PI_CONFLICT.search("π_synth = 0.3") and not PI_CONFLICT.search("π = 0.5")


def test_a3_metric_constants_match_docs():
    assert metrics.C_FA == 4 and metrics.C_MISS == 1 and metrics.PI_SYNTH == 0.3
    for p in (REPO / "CLAUDE.md", REPO / "docs" / "plan.md"):
        text = p.read_text()
        assert C_FA_DOC.search(text), f"{p.name}: C_FA = 4 not stated"
        assert PI_DOC.search(text), f"{p.name}: π_synth = 0.3 not stated"


@pytest.mark.parametrize("path", LIVING, ids=lambda p: str(p.relative_to(REPO)))
def test_a3_no_conflicting_prior(path):
    assert not PI_CONFLICT.search(path.read_text())


# --- A4 no submission-CSV wording ----------------------------------------------------------


@pytest.mark.parametrize("path", LIVING, ids=lambda p: str(p.relative_to(REPO)))
def test_a4_no_csv_wording(path):
    bad = [ln for ln in path.read_text().splitlines()
           if CSV_WORD.search(ln) and not any(a in ln for a in CSV_ALLOW)]  # fmt: skip
    assert not bad, bad


# --- A5 never-cut set ----------------------------------------------------------------------


def test_a5_never_cut_set_listed():
    cut = section((REPO / "docs" / "plan.md").read_text(), "Cut order")
    line = next((ln for ln in cut.splitlines() if ln.startswith("Never cut:")), None)
    assert line, "no 'Never cut:' line in the Cut order section"
    for word in ("TSV", "grouped", "SSL", "Docker", "README"):
        assert word in line, word


# --- A7 historical docs are frozen ---------------------------------------------------------


def _git_show(path: str) -> str | None:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        return None
    ok = subprocess.run(["git", "cat-file", "-e", f"{BASE_SHA}^{{commit}}"], cwd=REPO,
                        capture_output=True, check=False)  # fmt: skip
    if ok.returncode != 0:
        return None
    return subprocess.run(["git", "show", f"{BASE_SHA}:{path}"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout  # fmt: skip


@pytest.mark.parametrize("path", ["docs/scoping.md", "docs/master-doc.md"])
def test_a7_historical_docs_changed_only_by_banner(path):
    base = _git_show(path)
    if base is None:
        pytest.skip(f"git history with {BASE_SHA} not available (e.g. Docker or shallow clone)")
    lines = (REPO / path).read_text().splitlines(keepends=True)
    banners = [i for i, ln in enumerate(lines) if ln.startswith(BANNER_PREFIX)]
    assert len(banners) == 1, "expected exactly one historical banner line"
    del lines[banners[0]]
    assert "".join(lines) == base


def test_a7_first_handoff_untouched():
    path = "docs/handoffs/2026-09-25_first-two-hours-handoff.md"
    base = _git_show(path)
    if base is None:
        pytest.skip(f"git history with {BASE_SHA} not available")
    assert (REPO / path).read_text() == base


# --- A8 standing failure modes rewritten for software-only ---------------------------------


def test_a8_standing_failure_modes_cover_software_scope():
    text = (REPO / ".claude" / "skills" / "plan-review" / "SKILL.md").read_text()
    start = text.index("**HEARSAY standing failure modes**")
    block = text[start : text.index("2. Then specify:", start)]
    for pat in [r"launder|telephony", r"fold file|grouped", r"π|prior", r"shortcut", r"MP4"]:
        assert re.search(pat, block, re.IGNORECASE), pat
    assert "if unknown" not in block
