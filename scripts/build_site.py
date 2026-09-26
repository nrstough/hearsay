#!/usr/bin/env python3
"""Build the HEARSAY documentation site under docs/site/ (standard library only).

Run spec: docs/specs/2026-09-26_docs-site.md. Plan: docs/reports/2026-09-26_docs-site-plan.md.

    uv run python scripts/build_site.py                 # build docs/site/
    uv run python scripts/build_site.py --strict        # missing links/diagrams are errors
    uv run python scripts/build_site.py --check         # committed output == fresh build?
    uv run python scripts/build_site.py --check-citations
    uv run python scripts/build_site.py --extract-mermaid   # fenced blocks -> source/diagrams/from-docs/
    uv run python scripts/build_site.py --out <dir>     # build elsewhere (tests)

Inputs: the markdown corpus (README.md, CLAUDE.md, docs/**/*.md, .claude/**/*.md), the hand-written
sources under docs/site/source/ and the pre-rendered diagrams under docs/site/img/diagrams/ (rendered
by docs/site/source/render-diagrams.js; the manifest there records each source's hash).
Outputs: everything else under docs/site/. The build is deterministic: no timestamps, no git data,
sorted walks, UTF-8 with LF. Pages carry a content hash of their sources, never a commit.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import posixpath
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "docs" / "site"
SOURCE = SITE / "source"
DIAGRAM_DIR = SITE / "img" / "diagrams"
GITHUB = "https://github.com/nrstough/hearsay"
GITHUB_BLOB = GITHUB + "/blob/main/"
GITHUB_TREE = GITHUB + "/tree/main/"
MAC_PREFIX = "/Users/nathanstough/Projects/hearsay/"
GITIGNORED_ROOTS = ("models/", "outputs/", "data/", "weights/", "submissions/")
CORPUS_GLOBS = ("README.md", "CLAUDE.md", "docs/**/*.md", ".claude/**/*.md")
INPUT_DIRS = ("source", "img/diagrams")  # inside docs/site/, never generated or deleted
ALLOWED_TAGS = {
    "span", "p", "em", "strong", "sub", "sup", "code", "br", "table", "colgroup", "col",
    "thead", "tbody", "tr", "th", "td", "img",
}
ALLOWED_ATTRS = {"class", "style", "src", "width", "height", "alt", "colspan", "rowspan"}
ALLOWED_STYLE_PROPS = {"width", "max-width", "height", "border-radius", "text-align"}
VOID_TAGS = {"br", "col", "img"}
SECTIONS = ("story", "specs", "library", "about")
SECTION_TITLES = {
    "story": "The story",
    "specs": "Systems and subsystems",
    "library": "The library",
    "about": "About this site",
}
LIBRARY_GROUPS = (
    ("top", "Top-level documents"),
    ("reports", "Reports and plan reviews"),
    ("specs", "Run specs and audits"),
    ("handoffs", "Handoffs"),
    ("consults", "Consult records"),
    ("method", "Working method (skills and review rubrics)"),
)
NUM_ALLOW = [
    r"\b(?:19|20)\d\d\b",  # years
    r"\b\d{4}-\d{2}-\d{2}\b",  # dates
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b",  # times
    r"(?:§|section\s)\s*\d+(?:\.\d+)*",  # section numbers
    r"\b(?:Python|Mermaid|torchaudio|Playwright|Chromium|Node|uv|torch|v)\s?\d+(?:\.\d+)*\b",
    r"\bHGT\d+\b",
    r"\b[A-Za-z]\d{3,}\b",
]
NUM_RE = re.compile(r"(?<![\w.,/#-])(\d+(?:\.\d+)?%|\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})(?![\w.,%]|,\d)")


class BuildError(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def attr(text: str) -> str:
    return html.escape(text, quote=True)


def sha12(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:12]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def slugify(text: str) -> str:
    """GitHub's heading slug: lowercase; keep letters, digits, marks, spaces, hyphens and
    underscores; spaces to hyphens. Duplicates are handled by the caller."""
    text = re.sub(r"`", "", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    out = []
    for ch in text.lower():
        if ch.isalnum() or ch in " -_" or unicodedata.category(ch).startswith("M"):
            out.append(ch)
    return "".join(out).strip().replace(" ", "-")


def rel_url(target: str, page_url: str) -> str:
    """Relative link from page_url to target, both site-relative POSIX paths."""
    start = posixpath.dirname(page_url) or "."
    return posixpath.relpath(target, start)


# ----------------------------------------------------------------------------------------------
# corpus
# ----------------------------------------------------------------------------------------------


def corpus_files() -> list[str]:
    found: set[str] = set()
    for pattern in CORPUS_GLOBS:
        for p in REPO.glob(pattern):
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith("docs/site/"):
                continue
            if p.is_file():
                found.add(rel)
    files = sorted(found)
    if len(files) < 80:
        raise BuildError(f"corpus discovery found only {len(files)} files")
    return files


def library_slug(rel: str) -> str:
    p = rel
    if p == "README.md":
        return "readme"
    if p == "CLAUDE.md":
        return "claude-md"
    if p.startswith(".claude/skills/") and p.endswith("/SKILL.md"):
        return "skill-" + p.split("/")[2]
    if p.startswith(".claude/"):
        p = "claude-" + p[len(".claude/"):]
    elif p.startswith("docs/"):
        p = p[len("docs/"):]
    p = p.removesuffix(".md")
    p = re.sub(r"[^A-Za-z0-9]+", "-", p).strip("-").lower()
    return p


def library_group(rel: str) -> str:
    if rel.startswith("docs/reports/"):
        return "reports"
    if rel.startswith("docs/specs/"):
        return "specs"
    if rel.startswith("docs/handoffs/"):
        return "handoffs"
    if rel.startswith("docs/consults/"):
        return "consults"
    if rel.startswith(".claude/"):
        return "method"
    return "top"


def title_from_path(rel: str) -> str:
    name = posixpath.basename(rel)
    name = name.removesuffix(".md")
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(.*)", name)
    if m:
        date, rest = m.groups()
        kind = ""
        for suffix, label in (("-plan-review", "plan review"), ("-audit", "audit"), ("-plan", "plan")):
            if rest.endswith(suffix):
                rest, kind = rest[: -len(suffix)], label
                break
        rest = rest.replace("_", " ").replace("-", " ")
        return f"{date} {rest}" + (f" ({kind})" if kind else "")
    return name.replace("_", " ").replace("-", " ")


# ----------------------------------------------------------------------------------------------
# markdown converter
# ----------------------------------------------------------------------------------------------

FENCE_RE = re.compile(r"^(\s*)(`{3,})\s*(\S*)\s*$")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
HR_RE = re.compile(r"^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])( +|$)(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)*\|?\s*$")
HTML_BLOCK_RE = re.compile(r"^<(table|p|span|img|div|colgroup|thead|tbody|tr|/table|/span|/p)\b")
TAG_RE = re.compile(
    r"<(/?)([a-zA-Z][a-zA-Z0-9]*)((?:\s+[a-zA-Z-]+(?:=\"[^\"]*\"|='[^']*'|=[^\s>]+)?)*)\s*(/?)>"
)
ATTR_RE = re.compile(r"([a-zA-Z-]+)(?:=(\"[^\"]*\"|'[^']*'|[^\s>]+))?")
CODE_SPAN_RE = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)", re.DOTALL)
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
STRONG_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL)
STRONG_US_RE = re.compile(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)", re.DOTALL)
EM_STAR_RE = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])", re.DOTALL)
EM_US_RE = re.compile(r"(?<!\w)_(?=[^\s_])(.+?)(?<=[^\s_])_(?!\w)", re.DOTALL)
ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|$<>~\"])")
HARD_BREAK_RE = re.compile(r"(?:  |\\)\n")
TASK_RE = re.compile(r"^\[([ xX])\]\s+")
PH = "\x00"


class Converter:
    """Markdown to HTML for this corpus. `link` maps (href) -> (href, note) or None."""

    def __init__(self, link=None):
        self.link = link or (lambda href: (href, None))
        self.headings: list[tuple[int, str, str]] = []
        self.slugs: dict[str, int] = {}
        self.warnings: list[str] = []
        self.mermaid_blocks: list[str] = []
        self.mermaid_embed = None  # callable(text) -> html, set by the site builder
        self._ph: list[str] = []

    # --- block level ---------------------------------------------------------------------

    def convert(self, text: str) -> str:
        text = text.replace("\r\n", "\n")
        lines = text.split("\n")
        lines = self._strip_front_matter(lines)
        return self._blocks(lines)

    @staticmethod
    def _strip_front_matter(lines: list[str]) -> list[str]:
        if lines and lines[0].strip() == "---":
            for i in range(1, min(len(lines), 40)):
                if lines[i].strip() == "---":
                    return lines[i + 1:]
        return lines

    @staticmethod
    def front_matter(text: str) -> dict[str, str]:
        lines = text.replace("\r\n", "\n").split("\n")
        meta: dict[str, str] = {}
        if lines and lines[0].strip() == "---":
            for i in range(1, min(len(lines), 40)):
                if lines[i].strip() == "---":
                    break
                m = re.match(r"^([A-Za-z_-]+):\s*(.*)$", lines[i])
                if m:
                    meta[m.group(1)] = m.group(2).strip()
        return meta

    def _blocks(self, lines: list[str]) -> str:
        out: list[str] = []
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            m = FENCE_RE.match(line)
            if m:
                i = self._fence(lines, i, m, out)
                continue
            m = HEADING_RE.match(line)
            if m:
                out.append(self._heading(len(m.group(1)), m.group(2)))
                i += 1
                continue
            if HR_RE.match(line):
                out.append("<hr>")
                i += 1
                continue
            if re.match(r"^ {0,3}>", line):
                i = self._blockquote(lines, i, out)
                continue
            m = LIST_RE.match(line)
            if m and (m.group(3) or not m.group(4)):
                i = self._list(lines, i, out)
                continue
            if line.lstrip().startswith("|") and i + 1 < n and TABLE_SEP_RE.match(lines[i + 1]) \
                    and lines[i + 1].strip().startswith("|"):
                i = self._table(lines, i, out)
                continue
            if HTML_BLOCK_RE.match(line.strip()):
                i = self._html_block(lines, i, out)
                continue
            i = self._paragraph(lines, i, out)
        return "\n".join(out)

    def _fence(self, lines, i, m, out) -> int:
        indent, ticks, lang = len(m.group(1)), m.group(2), m.group(3)
        body: list[str] = []
        j = i + 1
        while j < len(lines):
            close = re.match(r"^\s*(`{3,})\s*$", lines[j])
            if close and len(close.group(1)) >= len(ticks):
                break
            body.append(lines[j][indent:] if lines[j][:indent].strip() == "" else lines[j].lstrip())
            j += 1
        code = "\n".join(body)
        if lang == "mermaid":
            self.mermaid_blocks.append(code)
            if self.mermaid_embed is not None:
                out.append(self.mermaid_embed(code))
            else:
                out.append(f'<pre class="mermaid-source"><code>{esc(code)}</code></pre>')
        else:
            cls = f' class="language-{attr(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{esc(code)}</code></pre>")
        return j + 1

    def _heading(self, level: int, text: str) -> str:
        base = slugify(text) or "section"
        count = self.slugs.get(base, 0)
        self.slugs[base] = count + 1
        slug = base if count == 0 else f"{base}-{count}"
        self.headings.append((level, self._plain(text), slug))
        return f'<h{level} id="{attr(slug)}">{self._inline(text)}</h{level}>'

    def _blockquote(self, lines, i, out) -> int:
        body: list[str] = []
        j = i
        while j < len(lines) and lines[j].strip():
            s = lines[j]
            if re.match(r"^ {0,3}>", s):
                s = re.sub(r"^ {0,3}> ?", "", s)
            body.append(s)
            j += 1
        out.append("<blockquote>\n" + self._blocks(body) + "\n</blockquote>")
        return j

    def _list(self, lines, i, out) -> int:
        first = LIST_RE.match(lines[i])
        base_indent = len(first.group(1))
        ordered = first.group(2)[0].isdigit()
        start = int(first.group(2)[:-1]) if ordered else 1
        items: list[list[str]] = []
        loose = False
        j = i
        n = len(lines)
        content_indent = 0
        while j < n:
            line = lines[j]
            m = LIST_RE.match(line)
            if m and len(m.group(1)) == base_indent and (m.group(2)[0].isdigit() == ordered):
                content_indent = base_indent + len(m.group(2)) + max(1, len(m.group(3)))
                items.append([m.group(4)])
                j += 1
                continue
            if not line.strip():
                # blank: continue only if the next non-blank line is indented or a sibling item
                k = j
                while k < n and not lines[k].strip():
                    k += 1
                if k >= n:
                    break
                nxt = lines[k]
                nm = LIST_RE.match(nxt)
                if len(nxt) - len(nxt.lstrip()) >= content_indent or (
                    nm and len(nm.group(1)) == base_indent
                    and (nm.group(2)[0].isdigit() == ordered)
                ):
                    if len(nxt) - len(nxt.lstrip()) >= content_indent or k > j:
                        loose = loose or (k > j and len(nxt) - len(nxt.lstrip()) >= content_indent) \
                            or (nm is not None and len(nm.group(1)) == base_indent)
                    items[-1].extend([""] * (k - j))
                    j = k
                    continue
                break
            indent = len(line) - len(line.lstrip())
            if indent >= content_indent:
                items[-1].append(line[content_indent:])
                j += 1
                continue
            if m and len(m.group(1)) < base_indent:
                break
            if m and len(m.group(1)) > base_indent:
                # a deeper item with less indent than the content column: treat as nested
                items[-1].append(line[base_indent:])
                j += 1
                continue
            # lazy continuation of a paragraph line
            prev = items[-1][-1] if items[-1] else ""
            if prev.strip() and not FENCE_RE.match(prev) and not HEADING_RE.match(line) \
                    and not HR_RE.match(line) and not line.lstrip().startswith("|"):
                items[-1].append(line.strip())
                j += 1
                continue
            break
        tag = "ol" if ordered else "ul"
        attrs = f' start="{start}"' if ordered and start != 1 else ""
        rendered = []
        for item in items:
            while item and not item[-1].strip():
                item.pop()
            task = TASK_RE.match(item[0]) if item else None
            if task:
                item[0] = TASK_RE.sub("", item[0], count=1)
            inner = self._blocks(item)
            if not loose:
                inner = re.sub(r"^<p>(.*?)</p>", r"\1", inner, count=1, flags=re.DOTALL)
            if task:
                box = "checked " if task.group(1).lower() == "x" else ""
                inner = f'<input type="checkbox" disabled {box}> ' + inner
            rendered.append(f"<li>{inner}</li>")
        out.append(f"<{tag}{attrs}>\n" + "\n".join(rendered) + f"\n</{tag}>")
        return j

    def _split_cells(self, row: str) -> list[str]:
        cells: list[str] = []
        cur: list[str] = []
        in_code = 0
        i = 0
        while i < len(row):
            ch = row[i]
            if ch == "\\" and i + 1 < len(row) and row[i + 1] == "|":
                cur.append("\\|")
                i += 2
                continue
            if ch == "`":
                run = len(row[i:]) - len(row[i:].lstrip("`"))
                if in_code == 0:
                    in_code = run
                elif in_code == run:
                    in_code = 0
                cur.append("`" * run)
                i += run
                continue
            if ch == "|" and in_code == 0:
                cells.append("".join(cur))
                cur = []
                i += 1
                continue
            cur.append(ch)
            i += 1
        cells.append("".join(cur))
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        return [c.strip() for c in cells]

    def _table(self, lines, i, out) -> int:
        header = self._split_cells(lines[i])
        aligns = []
        for c in self._split_cells(lines[i + 1]):
            c = c.strip()
            if c.startswith(":") and c.endswith(":"):
                aligns.append("center")
            elif c.endswith(":"):
                aligns.append("right")
            elif c.startswith(":"):
                aligns.append("left")
            else:
                aligns.append("")
        ncol = len(header)
        rows = []
        j = i + 2
        while j < len(lines) and lines[j].strip().startswith("|"):
            cells = self._split_cells(lines[j])
            if len(cells) > ncol:
                self.warnings.append(f"table row has {len(cells)} cells, header {ncol}: {lines[j][:60]}")
                cells = cells[:ncol]
            cells += [""] * (ncol - len(cells))
            rows.append(cells)
            j += 1

        def cell(tag, k, text):
            a = f' style="text-align:{aligns[k]}"' if k < len(aligns) and aligns[k] else ""
            return f"<{tag}{a}>{self._inline(text)}</{tag}>"

        parts = ["<div class=\"table-wrap\"><table>", "<thead><tr>"]
        parts.append("".join(cell("th", k, c) for k, c in enumerate(header)))
        parts.append("</tr></thead>")
        if rows:
            parts.append("<tbody>")
            for r in rows:
                parts.append("<tr>" + "".join(cell("td", k, c) for k, c in enumerate(r)) + "</tr>")
            parts.append("</tbody>")
        parts.append("</table></div>")
        out.append("\n".join(parts))
        return j

    def _html_block(self, lines, i, out) -> int:
        body = []
        j = i
        while j < len(lines) and lines[j].strip() and not FENCE_RE.match(lines[j]):
            body.append(lines[j])
            j += 1
        out.append(self._inline("\n".join(body)))
        return j

    def _paragraph(self, lines, i, out) -> int:
        body = [lines[i]]
        j = i + 1
        n = len(lines)
        while j < n:
            line = lines[j]
            if not line.strip():
                break
            if FENCE_RE.match(line) or HEADING_RE.match(line) or HR_RE.match(line):
                break
            m = LIST_RE.match(line)
            if m and m.group(3) and (not m.group(2)[0].isdigit() or m.group(2) in ("1.", "1)")):
                break
            if re.match(r"^ {0,3}>", line):
                break
            if line.lstrip().startswith("|") and j + 1 < n and TABLE_SEP_RE.match(lines[j + 1]) \
                    and lines[j + 1].strip().startswith("|"):
                break
            if HTML_BLOCK_RE.match(line.strip()):
                break
            body.append(line)
            j += 1
        out.append("<p>" + self._inline("\n".join(body)) + "</p>")
        return j

    # --- inline level --------------------------------------------------------------------

    def _stash(self, s: str) -> str:
        self._ph.append(s)
        return f"{PH}{len(self._ph) - 1}{PH}"

    def _restore(self, s: str) -> str:
        def rep(m):
            return self._ph[int(m.group(1))]

        prev = None
        while prev != s:
            prev = s
            s = re.sub(f"{PH}(\\d+){PH}", rep, s)
        return s

    def _plain(self, text: str) -> str:
        """Plain text of a heading (for titles and the search index)."""
        t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
        t = re.sub(r"[`*_]", "", t)
        t = re.sub(r"<[^>]+>", "", t)
        return t.strip()

    def _sanitize_tag(self, m) -> str | None:
        closing, name, attrs = m.group(1), m.group(2).lower(), m.group(3)
        if name not in ALLOWED_TAGS:
            return None
        if closing:
            return f"</{name}>"
        kept = []
        for am in ATTR_RE.finditer(attrs or ""):
            k, v = am.group(1).lower(), am.group(2)
            if k not in ALLOWED_ATTRS:
                continue
            v = (v or "").strip("\"'")
            if k == "src" and not v.startswith("data:"):
                return None
            if k == "style":
                props = []
                for decl in v.split(";"):
                    if ":" not in decl:
                        continue
                    pk, pv = decl.split(":", 1)
                    if pk.strip().lower() in ALLOWED_STYLE_PROPS and "url(" not in pv.lower():
                        props.append(f"{pk.strip()}:{pv.strip()}")
                v = ";".join(props)
                if not v:
                    continue
            kept.append(f' {k}="{attr(v)}"')
        return f"<{name}{''.join(kept)}>"

    def _inline(self, text: str) -> str:
        self._ph = []
        s = text
        # 1. code spans
        s = CODE_SPAN_RE.sub(lambda m: self._stash(f"<code>{esc(m.group(2).strip(' '))}</code>"), s)
        # 2. hard breaks
        s = HARD_BREAK_RE.sub(lambda m: self._stash("<br>") + "\n", s)
        # 3. allowlisted raw tags

        def tag(m):
            t = self._sanitize_tag(m)
            return self._stash(t if t is not None else esc(m.group(0)))

        s = TAG_RE.sub(tag, s)
        # 4. backslash escapes
        s = ESCAPE_RE.sub(lambda m: self._stash(esc(m.group(1))), s)
        # 5. escape what is left
        s = esc(s)
        # 6. images and links

        def image(m):
            alt, src = m.group(1), m.group(2)
            if src.startswith("data:"):
                href = src
            else:
                resolved = self.link(html.unescape(src))
                href = resolved[0] if resolved else src
            return self._stash(f'<img src="{attr(href)}" alt="{attr(alt)}">')

        s = IMAGE_RE.sub(image, s)

        def link(m):
            label, href = m.group(1), html.unescape(m.group(2))
            resolved = self.link(href)
            inner = self._emphasis(label)
            if resolved is None:
                return self._stash(f'<code title="not in the repository">{inner}</code>')
            url, note = resolved
            title = f' title="{attr(note)}"' if note else ""
            return self._stash(f'<a href="{attr(url)}"{title}>{inner}</a>')

        s = LINK_RE.sub(link, s)
        s = self._emphasis(s)
        return self._restore(s)

    def _emphasis(self, s: str) -> str:
        s = STRONG_RE.sub(lambda m: self._stash("<strong>") + m.group(1) + self._stash("</strong>"), s)
        s = STRONG_US_RE.sub(lambda m: self._stash("<strong>") + m.group(1) + self._stash("</strong>"), s)
        s = EM_STAR_RE.sub(lambda m: self._stash("<em>") + m.group(1) + self._stash("</em>"), s)
        s = EM_US_RE.sub(lambda m: self._stash("<em>") + m.group(1) + self._stash("</em>"), s)
        return s


def html_to_text(fragment: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.DOTALL)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------------------------
# site
# ----------------------------------------------------------------------------------------------


class Page:
    def __init__(self, url, title, section, lead="", sources=(), group="", order=0, description=""):
        self.url = url  # site-relative, e.g. "library/readme.html"
        self.title = title
        self.section = section
        self.lead = lead
        self.sources = list(sources)
        self.group = group
        self.order = order
        self.description = description
        self.heading = title
        self.body = ""
        self.headings: list[tuple[int, str, str]] = []
        self.text = ""
        self.hashes: list[tuple[str, str]] = []  # (source path, sha12)


class Site:
    def __init__(self, out: Path, strict: bool = False):
        self.out = out
        self.strict = strict
        self.warnings: list[str] = []
        self.corpus = corpus_files()
        self.corpus_set = set(self.corpus)
        self.lib_url = {rel: f"library/{library_slug(rel)}.html" for rel in self.corpus}
        if len(set(self.lib_url.values())) != len(self.lib_url):
            raise BuildError("library slugs collide")
        self.pages: list[Page] = []
        self.assets: dict[str, str] = {}  # repo-relative -> site-relative copy
        self.manifest = self._load_manifest()
        self.glossary = self._load_glossary()
        self.citations: list[tuple[str, str, str]] = []  # (value, source, page url)
        self.constants = self._metric_constants()
        self.anchor_cache: dict[str, dict[str, str]] = {}
        self.diagram_sources: dict[str, list[str]] = {}

    # --- inputs --------------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        p = DIAGRAM_DIR / "manifest.json"
        if not p.exists():
            return {}
        return json.loads(read(p))

    def _load_glossary(self) -> dict[str, dict]:
        p = SOURCE / "glossary.json"
        if not p.exists():
            return {}
        raw = json.loads(read(p))
        table: dict[str, dict] = {}
        for entry in raw:
            entry.setdefault("aliases", [])
            for key in [entry["term"]] + entry["aliases"]:
                table[key.lower()] = entry
        return table

    @staticmethod
    def _metric_constants() -> dict[str, float]:
        text = read(REPO / "src" / "hearsay" / "metrics.py")
        vals = {}
        for name in ("C_FA", "C_MISS", "PI_SYNTH"):
            m = re.search(rf"^{name}\s*=\s*([0-9.]+)", text, re.MULTILINE)
            if not m:
                raise BuildError(f"{name} not found in src/hearsay/metrics.py")
            vals[name] = float(m.group(1))
        c_fa, c_miss, pi = vals["C_FA"], vals["C_MISS"], vals["PI_SYNTH"]
        default = min(c_fa * (1 - pi), c_miss * pi)
        vals["DEFAULT"] = default
        vals["W_FA"] = c_fa * (1 - pi) / default
        vals["W_MISS"] = c_miss * pi / default
        return vals

    # --- link resolution -----------------------------------------------------------------

    def resolve(self, href: str, source_rel: str, page_url: str):
        """Return (url, note) for a markdown href found in source_rel, or None for a path that
        is not in the repository (gitignored). Records a warning for a missing target."""
        if re.match(r"^(https?:|mailto:|data:)", href):
            return href, None
        path, _, frag = href.partition("#")
        if not path:
            return "#" + frag, None
        line = ""
        repo_relative = False
        if path.startswith(MAC_PREFIX):
            path, repo_relative = path[len(MAC_PREFIX):], True
        m = re.match(r"^(.*?):(\d+)$", path)
        if m:
            path, line = m.group(1), m.group(2)
        src_dir = posixpath.dirname(source_rel)
        if repo_relative or path.startswith("/"):
            target = posixpath.normpath(path.lstrip("/"))
        else:
            target = posixpath.normpath(posixpath.join(src_dir, path))
        if target.startswith(".."):
            self._warn(f"link escapes the repository: {href} in {source_rel}")
            return href, None
        if target in self.corpus_set:
            url = rel_url(self.lib_url[target], page_url)
            return (url + ("#" + frag if frag else "")), None
        abs_target = REPO / target
        if abs_target.is_file() and abs_target.suffix.lower() in (".svg", ".png", ".jpg", ".jpeg", ".gif", ".mmd"):
            return rel_url(self.copy_asset(target), page_url), None
        for root in GITIGNORED_ROOTS:
            if target.startswith(root) and not abs_target.exists():
                return None
        if abs_target.is_dir():
            return GITHUB_TREE + target, "on GitHub"
        if abs_target.exists():
            anchor = f"#L{line}" if line else ("#" + frag if frag else "")
            return GITHUB_BLOB + target + anchor, "on GitHub"
        self._warn(f"link target not found: {href} (from {source_rel})")
        return href, "target not found"

    def copy_asset(self, rel: str) -> str:
        if rel not in self.assets:
            name = re.sub(r"[^A-Za-z0-9.]+", "-", posixpath.basename(rel))
            site_rel = f"img/{name}"
            if site_rel in self.assets.values():
                site_rel = f"img/{sha12(rel)}-{name}"
            self.assets[rel] = site_rel
            dst = self.out / site_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, dst)
        return self.assets[rel]

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        if self.strict:
            raise BuildError(msg)

    # --- diagrams ------------------------------------------------------------------------

    def _svg_size(self, site_rel: str) -> tuple[str, str]:
        p = self.out / site_rel
        if not p.exists():
            p = SITE / site_rel
        m = re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"', read(p)[:2000])
        if not m:
            return "", ""
        return str(round(float(m.group(1)))), str(round(float(m.group(2))))

    def diagram_cites(self, source_text: str) -> list[str]:
        cites = []
        for m in re.finditer(r"^%%\s*cites:\s*(.+)$", source_text, re.MULTILINE):
            cites += [c.strip() for c in m.group(1).split(",") if c.strip()]
        return cites

    def diagram_html(self, name: str, alt: str, page_url: str, source_text: str = "") -> str:
        if source_text and not name.startswith("doc-"):
            cites = self.diagram_cites(source_text)
            body = re.sub(r"^%%.*$", "", source_text, flags=re.MULTILINE)
            body = re.sub(r"#[0-9a-fA-F]{3,6}\b", " ", body)  # colours
            body = re.sub(r"[A-Za-z_]+\d+[A-Za-z_\d]*", " ", body)  # node ids such as H1, M5, f0
            for tok in NUM_RE.findall(body) + NUM_RE.findall(alt):
                self.citations.append((tok, "diagram:" + name, page_url))
            self.diagram_sources[name] = cites
        entry = self.manifest.get(name)
        if not entry:
            self._warn(f"diagram '{name}' has no rendered SVG pair (run render-diagrams.js)")
            return (f'<figure class="diagram diagram-missing"><pre class="mermaid-source"><code>'
                    f"{esc(source_text)}</code></pre><figcaption>{esc(alt)} (not rendered)</figcaption></figure>")
        light = f"img/diagrams/{name}-light.svg"
        dark = f"img/diagrams/{name}-dark.svg"
        for f in (light, dark):
            if not (SITE / f).exists():
                self._warn(f"diagram file missing: {f}")
        w, h = self._svg_size(light)
        dim = f' width="{w}" height="{h}"' if w else ""
        lurl, durl = rel_url(light, page_url), rel_url(dark, page_url)
        details = ""
        if source_text:
            details = (f"<details><summary>Mermaid source</summary><pre><code>"
                       f"{esc(source_text)}</code></pre></details>")
        return (f'<figure class="diagram">'
                f'<img class="diagram-img diagram-light" src="{lurl}" alt="{attr(alt)}"{dim}>'
                f'<img class="diagram-img diagram-dark" src="{durl}" alt="" aria-hidden="true"{dim}>'
                f'<figcaption>{esc(alt)} <a class="fullsize" href="{lurl}">Open full size</a></figcaption>'
                f"{details}</figure>")

    def mermaid_from_doc(self, code: str, page_url: str) -> str:
        name = "doc-" + sha12(code)
        return self.diagram_html(name, "Diagram from the document", page_url, code)

    # --- templates -----------------------------------------------------------------------

    def expand(self, fragment: str, page: Page) -> str:
        """Expand {{term}}, {{num}}, {{src}}, {{diagram}}, {{cost-panel}}, {{cost-explorer}}."""
        tip_counter = [0]
        used_terms: set[str] = set()

        def check_context(pos: int, tag: str) -> None:
            before = fragment[:pos]
            for t in ("a", "code", "pre"):
                opens = len(re.findall(rf"<{t}\b", before))
                closes = len(re.findall(rf"</{t}>", before))
                if opens > closes:
                    raise BuildError(f"{{{{{tag}}}}} inside <{t}> on {page.url}")

        def source_link(src: str) -> tuple[str, str]:
            path, _, frag = src.partition("#")
            if path in self.corpus_set:
                url = rel_url(self.lib_url[path], page.url) + ("#" + frag if frag else "")
                where = path + (f", section {frag}" if frag else "")
            elif (REPO / path).exists():
                anchor = "#" + frag if frag else ""
                url = GITHUB_BLOB + path + anchor
                where = path + (f" line {frag[1:]}" if frag.startswith("L") else "") + " (GitHub; assumes main is public)"
            else:
                raise BuildError(f"citation source not found: {src} on {page.url}")
            return url, where

        def rep(m):
            kind, body = m.group(1), m.group(2) or ""
            if kind == "term":
                check_context(m.start(), "term")
                name, _, display = body.partition("|")
                entry = self.glossary.get(name.strip().lower())
                if not entry:
                    raise BuildError(f"glossary has no entry for '{name}' ({page.url})")
                used_terms.add(entry["term"])
                tip_counter[0] += 1
                tid = f"tip-{tip_counter[0]}"
                definition = self.expand_inline_num(entry["definition"], page)
                return (f'<span class="term" tabindex="0" data-term="{attr(entry["term"])}" '
                        f'aria-describedby="{tid}">{esc(display or name)}'
                        f'<span class="tip-body" role="tooltip" id="{tid}">{definition}</span></span>')
            if kind == "num":
                value, _, src = body.partition("|")
                value, src = value.strip(), src.strip()
                if not src:
                    raise BuildError(f"{{{{num}}}} without a source: {value} on {page.url}")
                url, where = source_link(src)
                self.citations.append((value, src, page.url))
                return (f'<span class="num" data-src="{attr(src)}">{esc(value)}</span>'
                        f'<a class="src" href="{attr(url)}" title="{attr("Source: " + where)}">'
                        f'<span class="sr">source: {esc(where)}</span></a>')
            if kind == "src":
                src, _, label = body.partition("|")
                url, where = source_link(src.strip())
                stripped = re.sub(r"(?:lines?|§|L)\s*\d+(?:\.\d+)*(?:\s*[–-]\s*\d+)?", " ", label)
                for tok in re.findall(r"\d+(?:\.\d+)?%?", stripped):
                    self.citations.append((tok, src.strip(), page.url))
                return f'<a class="srclink" href="{attr(url)}" title="{attr(where)}">{esc(label or src)}</a>'
            if kind == "diagram":
                name, _, alt = body.partition("|")
                mmd = SOURCE / "diagrams" / f"{name.strip()}.mmd"
                text = read(mmd) if mmd.exists() else ""
                return self.diagram_html(name.strip(), alt.strip() or name.strip(), page.url, text)
            if kind == "cost-panel":
                return self.cost_panel(page)
            if kind == "cost-explorer":
                return self.cost_explorer(page)
            raise BuildError(f"unknown template tag {{{{{kind}}}}} on {page.url}")

        out = re.sub(r"\{\{(term|num|src|diagram|cost-panel|cost-explorer)(?::((?:[^{}]|\{[^{}]*\})*?))?\}\}", rep, fragment)
        if "{{" in out:
            bad = re.search(r"\{\{[^}]{0,40}", out)
            raise BuildError(f"unexpanded template tag on {page.url}: {bad.group(0)!r}")
        page.hashes  # noqa: B018 (kept for symmetry)
        return out

    def expand_inline_num(self, text: str, page: Page) -> str:
        """Glossary definitions may contain {{num}} and {{src}}; nothing else."""
        return self.expand(text, page) if "{{" in text else esc(text)

    # --- cost element --------------------------------------------------------------------

    def _cost_numbers(self) -> dict[str, str]:
        c = self.constants

        def fmt(x: float) -> str:
            s = f"{x:.4f}".rstrip("0").rstrip(".")
            return s if s else "0"

        return {
            "c_fa": fmt(c["C_FA"]), "c_miss": fmt(c["C_MISS"]), "pi": fmt(c["PI_SYNTH"]),
            "pi_real": fmt(1 - c["PI_SYNTH"]), "fa_term": fmt(c["C_FA"] * (1 - c["PI_SYNTH"])),
            "miss_term": fmt(c["C_MISS"] * c["PI_SYNTH"]), "default": fmt(c["DEFAULT"]),
            "w_fa": f"{c['W_FA']:.2f}", "w_miss": fmt(c["W_MISS"]),
            "expr": (f"({fmt(c['C_FA'])}*pfa*(1-{fmt(c['PI_SYNTH'])}) + "
                     f"{fmt(c['C_MISS'])}*pmiss*{fmt(c['PI_SYNTH'])}) / {fmt(c['DEFAULT'])}"),
        }

    def cost_panel(self, page: Page) -> str:
        n = self._cost_numbers()
        src = "src/hearsay/metrics.py#L29"
        url, where = GITHUB_BLOB + src.replace("#", "#"), "src/hearsay/metrics.py lines 29-31 (GitHub; assumes main is public)"
        self.citations += [(n["c_fa"], src, page.url), (n["pi"], src, page.url)]
        return f'''<!-- scan:off --><section class="cost-panel" aria-labelledby="cost-title">
<h2 id="cost-title">The cost function we are judged on</h2>
<p class="cost-formula" aria-label="normalized minimum detection cost">
<span class="cost-lhs">minDCF</span> <span class="cost-eq">=</span>
<span class="cost-term cost-fa"><span class="cost-w">{n["w_fa"]}</span> <span class="cost-var">×</span> <span class="cost-var">P<sub>FA</sub></span></span>
<span class="cost-plus">+</span>
<span class="cost-term cost-miss"><span class="cost-w">{n["w_miss"]}</span> <span class="cost-var">×</span> <span class="cost-var">P<sub>miss</sub></span></span>
</p>
<p class="cost-read">One real file called synthetic costs as much as <strong>{n["w_fa"]} missed fakes</strong>.
That single ratio shaped every decision in this system.</p>
<dl class="cost-legend">
<dt>P<sub>FA</sub></dt><dd>false-alarm rate: the share of real files scored above the threshold</dd>
<dt>P<sub>miss</sub></dt><dd>miss rate: the share of synthetic files scored below it</dd>
<dt>min</dt><dd>the best threshold is chosen by the scorer, so only the ranking of scores matters</dd>
</dl>
<p class="cost-derive">Built from the sponsor's weights <span class="num" data-src="{src}">C<sub>FA</sub> = {n["c_fa"]}</span>, C<sub>miss</sub> = {n["c_miss"]} and a prior of <span class="num" data-src="{src}">{n["pi"]}</span> synthetic
(about {n["pi_real"]} of files are real): cost = {n["c_fa"]} · P<sub>FA</sub> · {n["pi_real"]} + {n["c_miss"]} · P<sub>miss</sub> · {n["pi"]} = {n["fa_term"]} · P<sub>FA</sub> + {n["miss_term"]} · P<sub>miss</sub>,
divided by the best constant decision, {n["default"]}, so that 1.0 means "no better than always saying real" and 0.0 is perfect.
<a class="src" href="{attr(url)}" title="{attr("Source: " + where)}"><span class="sr">source: {esc(where)}</span></a></p>
</section><!-- scan:on -->'''

    def cost_explorer(self, page: Page) -> str:
        n = self._cost_numbers()
        return f'''<!-- scan:off --><div class="cost-explorer" data-expr="{attr(n["expr"])}">
<p>Move the two rates and watch the cost. The arithmetic is exactly the site's formula: <code>{esc(n["expr"])}</code> with <code>pfa</code> and <code>pmiss</code> as fractions.</p>
<div class="cost-controls">
<label>False-alarm rate P<sub>FA</sub> <output id="pfa-out">5%</output>
<input type="range" id="pfa" min="0" max="100" value="5" step="1" aria-describedby="pfa-out"></label>
<label>Miss rate P<sub>miss</sub> <output id="pmiss-out">20%</output>
<input type="range" id="pmiss" min="0" max="100" value="20" step="1" aria-describedby="pmiss-out"></label>
</div>
<p class="cost-result" aria-live="polite">Normalized cost: <output id="cost-out">—</output>
<span class="cost-note">(1.0 would be the constant "always real" answer)</span></p>
<noscript><p>Enable JavaScript for the live readout; the formula above is the whole computation.</p></noscript>
</div><!-- scan:on -->'''

    # --- pages ---------------------------------------------------------------------------

    def load_fragment_pages(self) -> None:
        for kind, section, folder in (("story", "story", "pages"), ("specs", "specs", "specs")):
            d = SOURCE / folder
            if not d.exists():
                continue
            for f in sorted(d.glob("*.frag")):
                text = read(f)
                m = re.match(r"^<!--\s*(\{.*?\})\s*-->\n", text, re.DOTALL)
                if not m:
                    raise BuildError(f"{f} lacks a JSON header comment")
                meta = json.loads(m.group(1))
                body = text[m.end():]
                slug = f.stem
                url = f"{slug}.html" if section == "story" else f"specs/{slug}.html"
                page = Page(url, meta["title"], section, meta.get("lead", ""), meta.get("sources", []),
                            meta.get("group", ""), meta.get("order", 0), meta.get("description", ""))
                page.heading = meta.get("heading", meta["title"])
                page.raw = body
                page.hashes = [(f.relative_to(REPO).as_posix(), sha12(text))]
                for s in page.sources:
                    if (REPO / s).exists():
                        page.hashes.append((s, sha12((REPO / s).read_bytes())))
                self.pages.append(page)

    def load_library_pages(self) -> None:
        for rel in self.corpus:
            text = read(REPO / rel)
            page = Page(self.lib_url[rel], "", "library", sources=[rel], group=library_group(rel))
            page.raw = text
            page.hashes = [(rel, sha12((REPO / rel).read_bytes()))]
            self.pages.append(page)

    def render_library_page(self, page: Page) -> None:
        rel = page.sources[0]
        conv = Converter(link=lambda href, r=rel, u=page.url: self.resolve(href, r, u))
        conv.mermaid_embed = lambda code, u=page.url: self.mermaid_from_doc(code, u)
        body = conv.convert(page.raw)
        self.warnings.extend(f"{rel}: {w}" for w in conv.warnings)
        meta = Converter.front_matter(page.raw)
        h1 = [h for h in conv.headings if h[0] == 1]
        if h1:
            page.title = h1[0][1]
        elif meta.get("name"):
            page.title = meta["name"]
        else:
            page.title = title_from_path(rel)
        page.description = meta.get("description", "")
        page.lead = f"Rendered from <code>{esc(rel)}</code>; the markdown file is the record."
        page.headings = conv.headings
        page.body = body
        page.text = html_to_text(body)

    def render_fragment_page(self, page: Page) -> None:
        if page.lead:
            page.lead = self.expand(page.lead, page)
        body = self.expand(page.raw, page)
        page.body = body
        page.headings = [(int(m.group(1)), html_to_text(m.group(3)), m.group(2))
                         for m in re.finditer(r'<h([2-4]) id="([^"]+)">(.*?)</h\1>', body, re.DOTALL)]
        page.text = html_to_text(body)

    def glossary_page(self) -> Page:
        page = Page("glossary.html", "Glossary", "story", "Every term that carries a tooltip on this site, defined once.",
                    ["docs/site/source/glossary.json"], order=90)
        entries = sorted({e["term"]: e for e in self.glossary.values()}.values(), key=lambda e: e["term"].lower())
        items = []
        for e in entries:
            d = self.expand_inline_num(e["definition"], page)
            aliases = f' <span class="aliases">(also: {esc(", ".join(e["aliases"]))})</span>' if e["aliases"] else ""
            items.append(f'<dt id="term-{attr(slugify(e["term"]))}">{esc(e["term"])}{aliases}</dt><dd>{d}</dd>')
        page.raw = "<dl class=\"glossary\">\n" + "\n".join(items) + "\n</dl>"
        page.body = page.raw
        page.text = html_to_text(page.body)
        gj = SOURCE / "glossary.json"
        page.hashes = [(gj.relative_to(REPO).as_posix(), sha12(gj.read_bytes()))] if gj.exists() else []
        return page

    def about_page(self) -> Page:
        page = Page("about.html", "About this site", "about",
                    "How these pages are made, what is hand-written and what is rendered, and the rules every number follows.")
        n = self._cost_numbers()
        routed = SOURCE / "routed.html"
        routed_html = read(routed) if routed.exists() else "<p>None recorded yet.</p>"
        page.raw = f'''<h2 id="what-is-here">What is here</h2>
<p>Four sections. <strong>The story</strong> and <strong>Systems and subsystems</strong> are written by hand for this site, in plain English, and every figure on them carries a source link. <strong>The library</strong> is every markdown document in the repository (README, CLAUDE.md, the plan, architecture, status, code map, the NSA brief, and every report, run spec, audit, handoff, consult record, working-method skill and review rubric), rendered as it is. The markdown files are the record; these pages are a copy with navigation, diagrams and explainers.</p>
<h2 id="numbers">The rule for numbers</h2>
<p>A number on a story or spec page is never typed from memory. Each one is written as a citation to a file and, where the file has one, a section; the build fails if the literal number is not found there. Decimals, percentages, thousands-grouped numbers and integers of 100 or more must be cited; small counts (ten detectors, 12 layers) may stand alone. The README wins where two documents disagree, and each disagreement is listed below rather than resolved here. Where no document states a number, the page leaves it out.</p>
<h2 id="hashes">Footers and versions</h2>
<p>Every page's footer names its source file(s) with the first 12 characters of the SHA-256 of that file's bytes. The hash identifies the exact version of the markdown a page was built from; a later commit that changes the file changes the hash, which is how the freshness check notices. Pages carry no commit hash, because a document edited in the same commit as the site cannot know that commit's hash before it exists. The report for this site (<code>docs/reports/2026-09-26_docs-site.md</code>) records the commit each build came from.</p>
<h2 id="build">How it is built</h2>
<p>One standard-library Python script, <code>scripts/build_site.py</code>, reads the markdown corpus and the hand-written sources under <code>docs/site/source/</code> and writes everything else under <code>docs/site/</code>. The output is committed, so nothing needs building to read the site; it works opened from a checkout (<code>docs/site/index.html</code>), from <code>python3 -m http.server</code> in <code>docs/site</code>, and as a GitHub Pages site from <code>main</code> / <code>docs</code>. There is no runtime dependency: no CDN, no webfont, no library. Diagrams are Mermaid sources pre-rendered to SVG in light and dark by <code>docs/site/source/render-diagrams.js</code> (Node, Playwright, Mermaid 11.17.2); the manifest beside them records each source's hash, and the build refuses a diagram whose source changed since it was rendered.</p>
<pre><code>uv run python scripts/build_site.py --strict          # rebuild
uv run python scripts/build_site.py --check           # is the committed site fresh?
uv run python scripts/build_site.py --check-citations # is every number where its citation says?
npm pack mermaid@11 && tar xzf mermaid-11*.tgz        # once, anywhere
node docs/site/source/render-diagrams.js --mermaid package/dist/mermaid.min.js
node docs/site/source/verify-pages.js --out /tmp/site-shots   # every page, two widths, two themes</code></pre>
<h2 id="cost-constants">The cost constants</h2>
<p>The cost function shown on the home page and the metric page is built from the constants in <code>src/hearsay/metrics.py</code> at build time: C<sub>FA</sub> = {n["c_fa"]}, C<sub>miss</sub> = {n["c_miss"]}, π<sub>synth</sub> = {n["pi"]}, normalizer {n["default"]}, so the weight on the false-alarm rate is {n["w_fa"]}. The interactive explorer evaluates <code>{esc(n["expr"])}</code>.</p>
<h2 id="not-here">What is not here</h2>
<p>No 404 page (it would live outside this site's folder). Code links point at GitHub and assume the repository is public and the branch merged to <code>main</code>; document links stay inside this site and work offline. Paths under <code>models/</code>, <code>outputs/</code>, <code>data/</code>, <code>weights/</code> and the submission TSVs are not in the repository and are shown as plain text.</p>
<h2 id="routed">Discrepancies found while building, routed to the maintainers</h2>
{routed_html}'''
        page.body = page.raw
        page.text = html_to_text(page.body)
        page.hashes = [("scripts/build_site.py", sha12(Path(__file__).read_bytes()))]
        return page

    def index_page(self, section: str) -> Page:
        title = SECTION_TITLES[section]
        url = f"{section}/index.html" if section in ("specs", "library") else f"{section}.html"
        page = Page(url, title, section)
        pages = [p for p in self.pages if p.section == section and p.url != url]
        if section == "library":
            page.lead = "Every markdown document in the repository, rendered as it is and grouped as the folders are."
            parts = []
            for gid, gtitle in LIBRARY_GROUPS:
                members = [p for p in pages if p.group == gid]
                if not members:
                    continue
                items = "\n".join(
                    f'<li><a href="{rel_url(p.url, url)}">{esc(p.title)}</a> <span class="path">{esc(p.sources[0])}</span></li>'
                    for p in sorted(members, key=lambda p: p.sources[0]))
                parts.append(f'<h2 id="{gid}">{esc(gtitle)}</h2>\n<ul class="doc-list">\n{items}\n</ul>')
            page.raw = "\n".join(parts)
        else:
            page.lead = "One page per system and subsystem, all with the same shape: what it is in plain words, what goes in and comes out, where it lives, every parameter with the line that sets it, status, numbers, known limits, the tests that pin it, and sources."
            groups: dict[str, list[Page]] = {}
            for p in sorted(pages, key=lambda p: (p.order, p.title)):
                groups.setdefault(p.group, []).append(p)
            parts = []
            for g, members in groups.items():
                items = "\n".join(
                    f'<li><a href="{rel_url(p.url, url)}">{esc(p.title)}</a><span class="lead-short"> {p.lead}</span></li>'
                    for p in members)
                parts.append(f'<h2 id="{attr(slugify(g) or "group")}">{esc(g)}</h2>\n<ul class="doc-list">\n{items}\n</ul>')
            page.raw = "\n".join(parts)
        page.body = page.raw
        page.text = html_to_text(page.body)
        return page

    # --- shell ---------------------------------------------------------------------------

    def nav_html(self, current: Page) -> str:
        parts = ['<nav class="site-nav" aria-label="Site">']
        for section in SECTIONS:
            index_url = {"story": "index.html", "specs": "specs/index.html",
                         "library": "library/index.html", "about": "about.html"}[section]
            cls = ' class="current-section"' if section == current.section else ""
            parts.append(f'<div class="nav-section"{cls}><a class="nav-title" href="{rel_url(index_url, current.url)}">{esc(SECTION_TITLES[section])}</a>')
            if section == current.section and section in ("story", "specs"):
                members = sorted((p for p in self.pages if p.section == section and not p.url.endswith("/index.html")),
                                 key=lambda p: (p.order, p.title))
                items = []
                for p in members:
                    here = ' aria-current="page"' if p.url == current.url else ""
                    items.append(f'<li><a href="{rel_url(p.url, current.url)}"{here}>{esc(p.title)}</a></li>')
                parts.append("<ul>" + "".join(items) + "</ul>")
            elif section == "library" and current.section == "library":
                items = []
                for gid, gtitle in LIBRARY_GROUPS:
                    items.append(f'<li><a href="{rel_url("library/index.html", current.url)}#{gid}">{esc(gtitle)}</a></li>')
                parts.append("<ul>" + "".join(items) + "</ul>")
            parts.append("</div>")
        parts.append("</nav>")
        return "\n".join(parts)

    def prev_next(self, page: Page) -> str:
        if page.section not in ("story", "specs"):
            return ""
        members = sorted((p for p in self.pages if p.section == page.section and not p.url.endswith("/index.html")),
                         key=lambda p: (p.order, p.title))
        urls = [p.url for p in members]
        if page.url not in urls:
            return ""
        i = urls.index(page.url)
        parts = ['<nav class="prev-next" aria-label="Previous and next">']
        if i > 0:
            parts.append(f'<a class="prev" rel="prev" href="{rel_url(members[i-1].url, page.url)}">← {esc(members[i-1].title)}</a>')
        if i < len(members) - 1:
            parts.append(f'<a class="next" rel="next" href="{rel_url(members[i+1].url, page.url)}">{esc(members[i+1].title)} →</a>')
        parts.append("</nav>")
        return "".join(parts)

    def shell(self, page: Page) -> str:
        root = rel_url("index.html", page.url)
        css = rel_url("site.css", page.url)
        js = rel_url("site.js", page.url)
        idx = rel_url("search-index.js", page.url)
        sources = ", ".join(f"<code>{esc(s)}</code> <span class=\"hash\">{h}</span>" for s, h in page.hashes) or "none"
        toc = ""
        hs = [h for h in page.headings if h[0] == 2]
        if len(hs) >= 3 and page.section != "library":
            toc = '<nav class="toc" aria-label="On this page"><h2>On this page</h2><ul>' + "".join(
                f'<li><a href="#{attr(s)}">{esc(t)}</a></li>' for _, t, s in hs) + "</ul></nav>"
        elif len(hs) >= 3:
            toc = '<details class="toc"><summary>On this page</summary><ul>' + "".join(
                f'<li><a href="#{attr(s)}">{esc(t)}</a></li>' for _, t, s in hs) + "</ul></details>"
        desc = attr(page.description or page.lead and html_to_text(page.lead)[:160] or page.title)
        lead = f'<p class="lead">{page.lead}</p>' if page.lead else ""
        own_h1 = page.section == "library" and re.search(r"<h1\b", page.body) is not None
        h1 = "" if own_h1 else f"<h1>{esc(page.heading)}</h1>"
        if own_h1:
            lead = f'<p class="lead lead-small">{page.lead}</p>' if page.lead else ""
        favicon = ("data:image/svg+xml," + "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
                   "%3Crect width='32' height='32' rx='6' fill='%2300254b'/%3E%3Cpath d='M7 20c3-9 6-9 9 0s6 9 9 0' "
                   "fill='none' stroke='%23ffb800' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E")
        return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(page.title)} · HEARSAY</title>
<meta name="description" content="{desc}">
<link rel="icon" href="{favicon}">
<link rel="stylesheet" href="{css}">
<script>try{{var t=localStorage.getItem("hearsay-theme");if(t)document.documentElement.setAttribute("data-theme",t);}}catch(e){{}}</script>
</head>
<body data-section="{page.section}" data-search-index="{idx}">
<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
<a class="brand" href="{root}"><span class="brand-mark" aria-hidden="true"></span>HEARSAY <span class="brand-sub">audio authentication, documented</span></a>
<div class="header-tools">
<form class="search" role="search" onsubmit="return false"><label class="sr" for="q">Search the site</label><input id="q" type="search" placeholder="Search…" autocomplete="off"></form>
<button type="button" class="theme-toggle" aria-pressed="false" aria-label="Switch to dark theme">Dark</button>
<a class="gh" href="{GITHUB}" title="Repository on GitHub (private until the team makes it public)">GitHub</a>
<label class="menu-toggle" for="nav-check">Menu</label>
</div>
</header>
<input type="checkbox" id="nav-check" class="nav-check" aria-label="Show the site navigation">
<div class="search-results" id="search-results" aria-live="polite" hidden></div>
<div class="layout">
<div id="site-nav" class="nav-wrap">{self.nav_html(page)}</div>
<main id="main" class="content">
<article>
{h1}
{lead}
{toc}
{page.body}
</article>
{self.prev_next(page)}
<footer class="page-footer">
<p>Generated from {sources}. The markdown is the record. <a href="{rel_url("about.html", page.url)}">About this site</a>.</p>
</footer>
</main>
</div>
<script src="{js}"></script>
</body>
</html>'''

    # --- search --------------------------------------------------------------------------

    def search_index(self) -> str:
        pages = sorted(self.pages, key=lambda p: p.url)
        records = []
        terms: dict[str, set[int]] = {}
        for pid, p in enumerate(pages):
            heads = " ".join(t for _, t, _ in p.headings)
            body = p.text
            if p.section == "library" and len(body) > 200_000:
                body = ""  # the three exports are indexed by title and headings only
            body = re.sub(r"[A-Za-z0-9+/=]{60,}", " ", body)  # base64 fragments
            records.append({"u": p.url, "t": p.title, "s": SECTION_TITLES[p.section], "h": heads[:300]})
            for weight_text in (p.title, heads, body):
                for tok in re.findall(r"[a-z0-9_]+", weight_text.lower()):
                    if 3 <= len(tok) <= 30 or (len(tok) == 2 and any(c.isdigit() for c in tok)):
                        terms.setdefault(tok, set()).add(pid)
        index = {"pages": records, "terms": {k: sorted(v) for k, v in sorted(terms.items())}}
        payload = json.dumps(index, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        payload = payload.replace("</script", "<\\/script")
        return "window.HEARSAY_INDEX=" + payload + ";"

    # --- build ---------------------------------------------------------------------------

    def generated_paths(self, root: Path) -> list[Path]:
        out = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if any(rel.startswith(d + "/") for d in INPUT_DIRS):
                continue
            out.append(p)
        return out

    def build(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        if self.out.resolve() == SITE.resolve():
            for p in self.generated_paths(SITE):
                p.unlink()
        else:
            # a fresh output directory still needs the input diagrams for the img links
            if DIAGRAM_DIR.exists():
                dst = self.out / "img" / "diagrams"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(DIAGRAM_DIR, dst)
        self.load_fragment_pages()
        self.load_library_pages()
        for p in list(self.pages):
            if p.section == "library":
                self.render_library_page(p)
            else:
                self.render_fragment_page(p)
        seen: dict[str, int] = {}
        for p in self.pages:
            if p.section == "library":
                seen[p.title] = seen.get(p.title, 0) + 1
        for p in self.pages:
            if p.section == "library" and seen.get(p.title, 0) > 1:
                p.title = f"{p.title} ({p.sources[0]})"
        if self.glossary:
            self.pages.append(self.glossary_page())
        self.pages.append(self.about_page())
        self.pages.append(self.index_page("specs"))
        self.pages.append(self.index_page("library"))
        for name in ("site.css", "site.js"):
            src = SOURCE / name
            if src.exists():
                write(self.out / name, read(src))
        if (REPO / "docs/img/architecture-flow.mmd").exists():
            self.copy_asset("docs/img/architecture-flow.mmd")
        for p in self.pages:
            write(self.out / p.url, self.shell(p))
        write(self.out / "search-index.js", self.search_index())
        # prune empty directories left by deleted pages
        for d in sorted((d for d in self.out.rglob("*") if d.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()

    # --- checks --------------------------------------------------------------------------

    def check_manifest(self) -> list[str]:
        problems = []
        sources: dict[str, str] = {}
        ddir = SOURCE / "diagrams"
        if ddir.exists():
            for f in sorted(ddir.rglob("*.mmd")):
                sources[f.stem] = hashlib.sha256(read(f).encode("utf-8")).hexdigest()
        for name, digest in sources.items():
            entry = self.manifest.get(name)
            if not entry:
                problems.append(f"diagram '{name}' is not rendered")
            elif entry.get("sha256") != digest:
                problems.append(f"diagram '{name}' changed since it was rendered")
            for theme in ("light", "dark"):
                if not (DIAGRAM_DIR / f"{name}-{theme}.svg").exists():
                    problems.append(f"diagram '{name}' lacks its {theme} SVG")
        return problems

    def section_text(self, path: str, frag: str) -> str | None:
        text = read(REPO / path)
        if not frag:
            return text
        if path not in self.anchor_cache:
            conv = Converter()
            conv.convert(text)
            self.anchor_cache[path] = {slug: plain for _, plain, slug in conv.headings}
        if frag not in self.anchor_cache[path]:
            return None
        # locate the heading line and cut to the next heading of the same or higher level
        lines = text.replace("\r\n", "\n").split("\n")
        conv = Converter()
        level = None
        start = None
        seen: dict[str, int] = {}
        for i, line in enumerate(lines):
            m = HEADING_RE.match(line)
            if not m:
                continue
            base = slugify(m.group(2)) or "section"
            c = seen.get(base, 0)
            seen[base] = c + 1
            slug = base if c == 0 else f"{base}-{c}"
            if start is None and slug == frag:
                start, level = i, len(m.group(1))
                continue
            if start is not None and len(m.group(1)) <= level:
                return "\n".join(lines[start:i])
        del conv
        return "\n".join(lines[start:]) if start is not None else None

    def check_citations(self) -> list[str]:
        problems = []
        for value, src, url in self.citations:
            variants = {value, value.replace(",", ""), value.replace(",", "_"), value.replace("−", "-")}
            if src.startswith("diagram:"):
                cites = self.diagram_sources.get(src[8:], [])
                if not cites:
                    problems.append(f"{url}: diagram {src[8:]} has numbers but no %% cites line")
                    continue
                found = False
                for c in cites:
                    cp, _, cf = c.partition("#")
                    text = self.section_text(cp, cf) if not cf.startswith("L") else None
                    if text is not None and any(v in text for v in variants):
                        found = True
                        break
                if not found:
                    problems.append(f"{url}: {value!r} (diagram {src[8:]}) not found in {cites}")
                continue
            path, _, frag = src.partition("#")
            if frag.startswith("L") and frag[1:].isdigit():
                lines = read(REPO / path).split("\n")
                n = int(frag[1:])
                window = "\n".join(lines[max(0, n - 4): n + 3])
                if not any(v in window for v in variants):
                    problems.append(f"{url}: {value!r} not within 3 lines of {src}")
                continue
            text = self.section_text(path, frag)
            if text is None:
                problems.append(f"{url}: section '{frag}' not found in {path}")
                continue
            if not any(v in text for v in variants):
                problems.append(f"{url}: {value!r} not found in {src}")
        # uncited numbers on story and spec pages
        allow = re.compile("|".join(NUM_ALLOW))
        for page in self.pages:
            if page.section not in ("story", "specs"):
                continue
            body = page.body + " " + page.lead
            body = re.sub(r"<pre\b.*?</pre>", " ", body, flags=re.DOTALL)
            body = re.sub(r"<code\b.*?</code>", " ", body, flags=re.DOTALL)
            body = re.sub(r'<span class="num"[^>]*>.*?</span>', " ", body, flags=re.DOTALL)
            body = re.sub(r'<a class="src"[^>]*>.*?</a>', " ", body, flags=re.DOTALL)
            body = re.sub(r'<span class="tip-body"[^>]*>.*?</span>', " ", body, flags=re.DOTALL)
            body = re.sub(r'<a class="srclink"[^>]*>.*?</a>', " ", body, flags=re.DOTALL)
            body = re.sub(r"<!-- scan:off -->.*?<!-- scan:on -->", " ", body, flags=re.DOTALL)
            body = re.sub(r"<figcaption>.*?</figcaption>", " ", body, flags=re.DOTALL)
            body = re.sub(r'<dl class="glossary">.*?</dl>', " ", body, flags=re.DOTALL)
            body = re.sub(r"<[^>]+>", " ", body)
            body = allow.sub(" ", body)
            for m in NUM_RE.finditer(body):
                ctx = body[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
                problems.append(f"{page.url}: uncited number {m.group(1)!r} in '…{ctx}…'")
        return problems


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def extract_mermaid() -> int:
    dst = SOURCE / "diagrams" / "from-docs"
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for rel in corpus_files():
        conv = Converter()
        conv.convert(read(REPO / rel))
        for code in conv.mermaid_blocks:
            name = "doc-" + sha12(code)
            write(dst / f"{name}.mmd", code)
            count += 1
    print(f"extracted {count} mermaid block(s) to {dst.relative_to(REPO).as_posix()}")
    return 0


def run_check() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "site"
        site = Site(out)
        site.build()
        problems = site.check_manifest()
        fresh = {p.relative_to(out).as_posix(): p.read_bytes() for p in site.generated_paths(out)}
        committed = {p.relative_to(SITE).as_posix(): p.read_bytes() for p in site.generated_paths(SITE)}
        for rel in sorted(set(fresh) | set(committed)):
            if rel not in committed:
                problems.append(f"missing from docs/site: {rel}")
            elif rel not in fresh:
                problems.append(f"stale file in docs/site: {rel}")
            elif fresh[rel] != committed[rel]:
                problems.append(f"differs: {rel}")
    for p in problems:
        print("CHECK:", p)
    print("check OK" if not problems else f"check FAILED ({len(problems)} problem(s))")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=SITE)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-citations", action="store_true")
    ap.add_argument("--extract-mermaid", action="store_true")
    args = ap.parse_args(argv)
    if args.extract_mermaid:
        return extract_mermaid()
    if args.check:
        return run_check()
    site = Site(args.out, strict=args.strict)
    site.build()
    for w in site.warnings:
        print("WARN:", w)
    manifest_problems = site.check_manifest()
    for p in manifest_problems:
        print("DIAGRAM:", p)
    if args.strict and manifest_problems:
        print("strict build failed")
        return 1
    n_lib = sum(1 for p in site.pages if p.section == "library")
    n_specs = sum(1 for p in site.pages if p.section == "specs")
    n_story = sum(1 for p in site.pages if p.section == "story")
    print(f"built {len(site.pages)} pages ({n_story} story, {n_specs} specs, {n_lib} library) "
          f"into {args.out}; {len(site.warnings)} warning(s)")
    if args.check_citations:
        problems = site.check_citations()
        for p in problems:
            print("CITE:", p)
        print("citations OK" if not problems else f"citations FAILED ({len(problems)} problem(s))")
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
