"""Tests for the documentation-site generator (run spec docs/specs/2026-09-26_docs-site.md).

Hermetic: every test builds into a temporary directory through one session-scoped fixture, or
exercises the converter on fixtures. Nothing here renders diagrams or opens a browser. The two
staleness checks (committed site == fresh build; the home-page numbers against the live README)
run only under HEARSAY_SITE_CHECK=1, so other lanes' markdown commits cannot turn the shared
suite red (plan amendment B1).
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_site.py"
SITE = REPO / "docs" / "site"
SOURCE = SITE / "source"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
STALE = pytest.mark.skipif(not os.environ.get("HEARSAY_SITE_CHECK"), reason="set HEARSAY_SITE_CHECK=1 to compare against the committed site")


def load_module():
    spec = importlib.util.spec_from_file_location("build_site", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def bs():
    return load_module()


@pytest.fixture(scope="session")
def built(bs, tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    site = bs.Site(out, strict=True)
    site.build()
    return site


@pytest.fixture(scope="session")
def pages(built):
    out = built.out
    return {p.relative_to(out).as_posix(): p.read_text(encoding="utf-8")
            for p in built.generated_paths(out) if p.suffix == ".html"}


def conv(bs, text, link=None):
    c = bs.Converter(link=link)
    return c.convert(text), c


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            self.errors.append(f"unexpected </{tag}> at {self.getpos()}")
        else:
            self.stack.pop()

    def check(self, html):
        self.feed(html)
        self.close()
        self.errors += [f"unclosed <{t}> at {pos}" for t, pos in self.stack]
        return self.errors


# --- converter: headings, slugs, anchors -------------------------------------------------------


def test_heading_levels_and_github_slugs(bs):
    html, c = conv(bs, "# One `code` here\n## Two: Q&A → done\n### Three\n#### Four\n##### Five\n###### Six\n")
    assert '<h1 id="one-code-here">' in html
    assert '<h2 id="two-qa--done">' in html
    assert [h[0] for h in c.headings] == [1, 2, 3, 4, 5, 6]


def test_duplicate_headings_get_suffixes(bs):
    html, _ = conv(bs, "## Scorecard\n\ntext\n\n## Scorecard\n")
    assert 'id="scorecard"' in html and 'id="scorecard-1"' in html


def test_combining_mark_kept_in_slug(bs):
    assert bs.slugify("A. How wild is the test set? (λ̂)") == "a-how-wild-is-the-test-set-λ̂"


def test_no_setext_headings(bs):
    html, _ = conv(bs, "some text\n---\n")
    assert "<h2" not in html and "<hr>" in html


def test_front_matter_stripped_and_read(bs):
    text = "---\nname: p3\ndescription: Execute a plan\n---\n\n# Title\n"
    html, _ = conv(bs, text)
    assert "name: p3" not in html and "<h1" in html
    assert bs.Converter.front_matter(text) == {"name": "p3", "description": "Execute a plan"}


# --- converter: lists ----------------------------------------------------------------------------


def test_nested_mixed_lists_three_deep(bs):
    text = "1. first\n   - inner `code` and [a link](https://example.com/x)\n     1. deepest\n2. second\n"
    html, _ = conv(bs, text)
    assert html.count("<ol") == 2 and html.count("<ul>") == 1
    assert "<code>code</code>" in html and 'href="https://example.com/x"' in html
    assert "deepest" in html.split("<ul>")[1]


def test_ordered_list_start_and_task_items(bs):
    html, _ = conv(bs, "3. three\n4. four\n")
    assert '<ol start="3">' in html
    html, _ = conv(bs, "- [ ] todo\n- [x] done\n")
    assert 'type="checkbox" disabled >' in html and "checked" in html


def test_fence_inside_list_item(bs):
    text = "1. Run:\n   ```bash\n   # not a heading\n   echo hi | cat\n   ```\n2. Next\n"
    html, _ = conv(bs, text)
    assert '<pre><code class="language-bash"># not a heading\necho hi | cat</code></pre>' in html
    assert "<h1" not in html and html.count("<li>") == 2


def test_table_inside_list_item(bs):
    text = "4. **Item**, then a table:\n\n   | A | B |\n   |---|---|\n   | 1 | 2 |\n\n5. next\n"
    html, _ = conv(bs, text)
    assert "<table>" in html and html.count("<li>") == 2


# --- converter: tables ---------------------------------------------------------------------------


def test_table_alignment_escaped_pipes_and_code_pipes(bs):
    text = "| A | B | C |\n|:--|---:|---|\n| x \\| y | `a|b` | z |\n"
    html, _ = conv(bs, text)
    assert 'style="text-align:left"' in html and 'style="text-align:right"' in html
    row = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)[1]
    assert len(re.findall(r"<td", row)) == 3
    assert "x | y" in row and "<code>a|b</code>" in row


def test_table_short_rows_padded_and_long_rows_warned(bs):
    text = "| A | B | C |\n|---|---|---|\n| 1 |\n| 1 | 2 | 3 | 4 |\n"
    html, c = conv(bs, text)
    for row in re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)[1:]:
        assert len(re.findall(r"<td", row)) == 3
    assert any("4 cells" in w for w in c.warnings)


def test_readme_numbers_table_has_equal_columns(bs):
    text = (REPO / "README.md").read_text(encoding="utf-8")
    html, _ = conv(bs, text)
    for table in re.findall(r"<table>.*?</table>", html, re.DOTALL):
        counts = {len(re.findall(r"<t[dh]\b", r)) for r in re.findall(r"<tr>(.*?)</tr>", table, re.DOTALL)}
        assert len(counts) == 1, counts


# --- converter: fences and inline -----------------------------------------------------------------


def test_fence_contents_verbatim_and_escaped(bs):
    text = "```text\n# heading?\n| not | a table |\n**not bold** <tag> & amp\n```\n"
    html, _ = conv(bs, text)
    assert "# heading?" in html and "<h1" not in html
    assert "**not bold** &lt;tag&gt; &amp; amp" in html and "<table" not in html


def test_math_fence_with_leading_space_renders_as_pre(bs):
    html, _ = conv(bs, "``` math\nx^2\n```\n")
    assert '<pre><code class="language-math">x^2</code></pre>' in html


def test_inline_code_escapes_and_keeps_pipes(bs):
    html, _ = conv(bs, "Header `filename<TAB>cm-score` and `a|b` and `x & y`.\n")
    assert "<code>filename&lt;TAB&gt;cm-score</code>" in html and "<code>a|b</code>" in html
    assert "<code>x &amp; y</code>" in html


def test_intra_word_underscores_are_not_emphasis(bs):
    html, _ = conv(bs, "grad_tts + unit_speech and P_FA and nsa_folds.csv but _real emphasis_ here.\n")
    assert "grad_tts + unit_speech" in html and "P_FA" in html and "nsa_folds.csv" in html
    assert "<em>real emphasis</em>" in html


def test_emphasis_styles(bs):
    html, _ = conv(bs, "**bold** and *it* and __post_init__ and ***x*** stays.\n")
    assert "<strong>bold</strong>" in html and "<em>it</em>" in html and "<strong>post_init</strong>" in html


def test_backslash_escapes_and_currency(bs):
    html, _ = conv(bs, "0.301\\* and \\# and ~$0.70 ≈ $7 total.\n")
    assert "0.301*" in html and "#" in html and "$0.70 ≈ $7" in html and "<em>" not in html


def test_placeholder_tags_in_prose_are_escaped(bs):
    html, _ = conv(bs, "models/<prefix>_<stamp> and <X> and <changing Y>.\n")
    assert "&lt;prefix&gt;" in html and "&lt;X&gt;" in html and "&lt;changing Y&gt;" in html


def test_allowlisted_raw_html_passes_and_others_escape(bs):
    text = '<span class="math inline">x</span> <sub>2</sub> <script>bad()</script> <img src="http://e/x.png"> <img src="data:image/png;base64,AAAA" style="max-width:100%;background:url(http://e)">'
    html, _ = conv(bs, text)
    assert '<span class="math inline">x</span>' in html and "<sub>2</sub>" in html
    assert "&lt;script&gt;" in html and '&lt;img src="http://e/x.png"&gt;' in html
    assert 'src="data:image/png;base64,AAAA"' in html and "url(" not in html and "max-width:100%" in html


def test_hard_breaks_blockquote_lazy_continuation_and_rule(bs):
    html, _ = conv(bs, "line one  \nline two\\\nline three\n\n> quoted\ncontinued\n\n***\n")
    assert html.count("<br>") == 2
    assert "<blockquote>" in html and "continued" in html.split("</blockquote>")[0]
    assert "<hr>" in html


def test_unicode_survives(bs):
    html, _ = conv(bs, "λ̂ = 0.51 · × ± → − ‖ ≥ † ‡\n")
    for ch in "λ̂·×±→−‖≥†‡":
        assert ch in html


def test_code_span_across_line_break_joined(bs):
    html, _ = conv(bs, "run `try again at <date>\n<time>` later\n")
    assert "<code>try again at &lt;date&gt;\n&lt;time&gt;</code>" in html


def test_whole_corpus_converts_cleanly(bs):
    files = bs.corpus_files()
    assert len(files) >= 80
    for rel in files:
        html, _ = conv(bs, (REPO / rel).read_text(encoding="utf-8"))
        errs = Balance().check(html)
        assert not errs, (rel, errs[:3])
        text = re.sub(r"<(pre|code)\b.*?</\1>", "", html, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        assert "**" not in text, rel
        assert not re.search(r"(?m)^\s*```", text), rel
        assert not re.search(r"\]\([^)]*\.md\)", text), rel
        assert not re.search(r"&(?!(#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);)", text), rel


def test_converter_is_fast_on_the_exports(bs):
    biggest = sorted(bs.corpus_files(), key=lambda r: (REPO / r).stat().st_size)[-2:]
    t0 = time.perf_counter()
    for rel in biggest:
        conv(bs, (REPO / rel).read_text(encoding="utf-8"))
    assert time.perf_counter() - t0 < 5.0


# --- links ---------------------------------------------------------------------------------------


def test_link_rewriting_forms(built):
    r = built.resolve
    assert r("docs/STATUS.md", "README.md", "library/readme.html") == ("status.html", None)
    assert r("../CLAUDE.md", "docs/STATUS.md", "library/status.html") == ("claude-md.html", None)
    assert r("../reports/2026-09-26_cpu-detectors.md", "docs/handoffs/x.md", "library/x.html")[0].endswith("reports-2026-09-26-cpu-detectors.html")
    assert r("CLAUDE.md#ai-use-disclosure", "README.md", "library/readme.html") == ("claude-md.html#ai-use-disclosure", None)
    assert r("#numbers", "README.md", "library/readme.html") == ("#numbers", None)


def test_code_and_directory_links_go_to_github(built):
    url, _ = built.resolve("../../src/hearsay/hc_v4.py", "docs/handoffs/x.md", "library/x.html")
    assert url.startswith("https://github.com/nrstough/hearsay/blob/main/src/hearsay/hc_v4.py")
    url, _ = built.resolve("docs/reports/", "README.md", "library/readme.html")
    assert url.startswith("https://github.com/nrstough/hearsay/tree/main/docs/reports")


def test_mac_absolute_paths_are_rewritten(built):
    url, _ = built.resolve("/Users/nathanstough/Projects/hearsay/scripts/m5_assemble.py:58", "docs/specs/x.md", "library/x.html")
    assert url == "https://github.com/nrstough/hearsay/blob/main/scripts/m5_assemble.py#L58"


def test_gitignored_targets_are_not_links(built):
    assert built.resolve("models/fusion_v2/constants.json", "README.md", "library/readme.html") is None


def test_external_links_untouched(built):
    for href in ("https://docs.astral.sh/uv/", "mailto:x@y.z", "data:image/png;base64,AAAA"):
        assert built.resolve(href, "README.md", "library/readme.html") == (href, None)


def test_missing_target_is_reported_and_strict_raises(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=False)
    site.resolve("docs/does-not-exist.md", "README.md", "library/readme.html")
    assert any("not found" in w for w in site.warnings)
    strict = bs.Site(tmp_path / "t", strict=True)
    with pytest.raises(bs.BuildError):
        strict.resolve("docs/does-not-exist.md", "README.md", "library/readme.html")


def test_images_and_mmd_are_copied(built, pages):
    assert (built.out / "img" / "architecture-flow.svg").exists()
    assert (built.out / "img" / "architecture-flow.mmd").exists()
    assert 'src="../img/architecture-flow.svg"' in pages["library/readme.html"]


def test_every_internal_href_resolves_with_exact_case(built, pages):
    for rel, html in pages.items():
        here = (built.out / rel).parent
        scan = re.sub(r"<(pre|code)\b.*?</\1>", "", html, flags=re.DOTALL)
        for href in re.findall(r'(?<![\w-])(?:href|src)="([^"#]+)', scan):
            if re.match(r"^(https?:|mailto:|data:)", href):
                continue
            assert not href.startswith("/"), (rel, href)
            assert not href.endswith("/"), (rel, href)
            target = (here / href).resolve()
            assert target.exists(), (rel, href)
            assert target.name in os.listdir(target.parent), (rel, href)


# --- generator invariants ------------------------------------------------------------------------


def test_page_counts_and_sections(built):
    sections = {}
    for p in built.pages:
        sections[p.section] = sections.get(p.section, 0) + 1
    assert sections["library"] >= 80
    assert sections["story"] >= 9
    assert sections["about"] == 1


def test_every_page_has_shell_invariants(pages):
    for rel, html in pages.items():
        assert html.startswith("<!doctype html>"), rel
        assert '<html lang="en">' in html and '<meta charset="utf-8">' in html, rel
        assert '<meta name="viewport"' in html and "<title>" in html, rel
        assert html.count("<h1") == 1, rel
        assert '<a class="skip" href="#main">' in html, rel
        assert "{{" not in re.sub(r"<(pre|code)\b.*?</\1>", "", html, flags=re.DOTALL), rel
        assert not Balance().check(html), rel


def test_ids_unique_per_page(pages):
    for rel, html in pages.items():
        ids = re.findall(r' id="([^"]+)"', html)
        dup = {i for i in ids if ids.count(i) > 1}
        assert not dup, (rel, dup)


def test_no_external_scripts_styles_modules_or_fetch(pages, built):
    for rel, html in pages.items():
        for m in re.finditer(r'<(script|link)\b[^>]*>', html):
            tag = m.group(0)
            src = re.search(r'(?:src|href)="([^"]+)"', tag)
            if src:
                assert not src.group(1).startswith("http"), (rel, tag)
            assert 'type="module"' not in tag, rel
        for script in re.findall(r"<script\b[^>]*>(.*?)</script>", html, re.DOTALL):
            assert "fetch(" not in script and "XMLHttpRequest" not in script, rel
    js = (built.out / "site.js").read_text(encoding="utf-8")
    assert "fetch(" not in js and "import " not in js and "http://" not in js and "https://" not in js
    css = (built.out / "site.css").read_text(encoding="utf-8")
    assert "http" not in css and "@import" not in css


def test_no_underscore_or_ignored_path_components(built):
    bad_dirs = {"build", "dist", "models", "outputs", "node_modules", ".next", "__pycache__"}
    for p in built.generated_paths(built.out):
        rel = p.relative_to(built.out)
        assert not any(part.startswith(("_", ".")) for part in rel.parts), rel
        assert not any(part in bad_dirs for part in rel.parts), rel
        assert p.suffix not in {".log", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".pyc"}, rel


def test_footer_content_hash_is_sha256_prefix(built, pages, bs):
    for rel in ("README.md", "CLAUDE.md", "docs/plan.md"):
        page = next(p for p in built.pages if p.section == "library" and p.sources == [rel])
        expected = bs.sha12((REPO / rel).read_bytes())
        assert page.hashes == [(rel, expected)]
        assert expected in pages[page.url]
    changed = bs.sha12((REPO / "README.md").read_bytes() + b"x")
    assert changed != bs.sha12((REPO / "README.md").read_bytes())


def test_no_git_hash_or_head_in_pages(pages):
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
    for rel, html in pages.items():
        footer = html.split('<footer class="page-footer">')[1]
        assert "HEAD" not in footer, rel
        if head:
            assert head not in footer, rel


def test_nav_lists_every_story_and_spec_page(built, pages):
    story = [p for p in built.pages if p.section == "story"]
    home = pages["index.html"]
    for p in story:
        assert f'href="{p.url}"' in home, p.url
    specs_index = pages["specs/index.html"]
    for p in built.pages:
        if p.section == "specs" and p.url != "specs/index.html":
            assert f'href="{p.url.split("/")[-1]}"' in specs_index, p.url


def test_library_index_lists_every_document(built, pages):
    index = pages["library/index.html"]
    for p in built.pages:
        if p.section == "library":
            assert f'href="{p.url.split("/")[-1]}"' in index, p.url


def test_library_titles_non_empty_and_unique(built):
    titles = [p.title for p in built.pages if p.section == "library"]
    assert all(t.strip() for t in titles)
    assert len(set(titles)) == len(titles), [t for t in titles if titles.count(t) > 1]


def test_library_title_rules(built):
    by_source = {p.sources[0]: p.title for p in built.pages if p.section == "library" and p.sources}
    assert by_source["README.md"] == "HEARSAY (README.md)" and by_source["CLAUDE.md"] == "HEARSAY (CLAUDE.md)"
    assert by_source[".claude/skills/zoom-out/SKILL.md"] == "zoom-out"
    assert "audit" in by_source["docs/specs/2026-09-26_k-docker-image-audit.md"]


def test_prev_next_chain(built, pages):
    story = sorted((p for p in built.pages if p.section == "story"), key=lambda p: (p.order, p.title))
    for a, b in itertools.pairwise(story):
        assert f'rel="next" href="{b.url}"' in pages[a.url]
        assert f'rel="prev" href="{a.url}"' in pages[b.url]


def test_build_is_deterministic(bs, tmp_path):
    outs = []
    for name in ("a", "b"):
        site = bs.Site(tmp_path / name, strict=True)
        site.build()
        outs.append({p.relative_to(site.out).as_posix(): p.read_bytes() for p in site.generated_paths(site.out)})
    assert outs[0].keys() == outs[1].keys()
    assert all(outs[0][k] == outs[1][k] for k in outs[0])


def test_check_detects_a_stale_file(bs, tmp_path, monkeypatch):
    site = bs.Site(tmp_path / "s", strict=True)
    site.build()
    fresh = {p.relative_to(site.out).as_posix(): p.read_bytes() for p in site.generated_paths(site.out)}
    (site.out / "index.html").write_bytes(b"stale")
    stale = {p.relative_to(site.out).as_posix(): p.read_bytes() for p in site.generated_paths(site.out)}
    assert fresh != stale and fresh["index.html"] != stale["index.html"]


@STALE
def test_committed_site_is_fresh(bs):
    assert bs.run_check() == 0


# --- diagrams ------------------------------------------------------------------------------------


def test_manifest_covers_every_mermaid_source(built):
    assert not built.check_manifest()


def test_rendered_svgs_are_clean(built):
    svgs = sorted((SITE / "img" / "diagrams").glob("*.svg"))
    assert len(svgs) >= 12
    for svg in svgs:
        text = svg.read_text(encoding="utf-8")
        assert 'viewBox="' in text and re.search(r'<svg[^>]* width="\d+" height="\d+"', text), svg.name
        assert 'width="100%"' not in text and "<script" not in text and "<foreignObject" not in text, svg.name
        assert "@import" not in text and 'href="http' not in text and "url(http" not in text and "<image" not in text, svg.name


def test_light_and_dark_svg_ids_disjoint():
    light = (SITE / "img" / "diagrams" / "pipeline-light.svg").read_text(encoding="utf-8")
    dark = (SITE / "img" / "diagrams" / "pipeline-dark.svg").read_text(encoding="utf-8")
    ids_l = set(re.findall(r' id="([^"]+)"', light))
    ids_d = set(re.findall(r' id="([^"]+)"', dark))
    assert ids_l and ids_d and not (ids_l & ids_d)


def test_fusion_diagram_matches_the_readme_rule():
    src = (SOURCE / "diagrams" / "fusion-arith.mmd").read_text(encoding="utf-8")
    for token in ("0.6", "0.2", "rank(M5)"):
        assert token in src
    pipeline = (SOURCE / "diagrams" / "pipeline.mmd").read_text(encoding="utf-8").replace("−", "-")
    assert "-3" in pipeline and "0.5" in pipeline
    readme = (REPO / "README.md").read_text(encoding="utf-8").replace("−", "-")
    assert "0.6 × rank(M1b) + 0.2 × rank(handcrafted) + 0.2 × rank(M5)" in readme
    assert re.search(r"margin is below -3", readme)


def test_diagram_embed_has_both_images_and_full_size_link(pages):
    html = pages["index.html"]
    assert 'class="diagram-img diagram-light" src="img/diagrams/pipeline-light.svg"' in html
    assert 'class="diagram-img diagram-dark" src="img/diagrams/pipeline-dark.svg"' in html
    assert 'class="fullsize" href="img/diagrams/pipeline-light.svg"' in html


def test_missing_diagram_renders_source_never_cdn(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=False)
    page = bs.Page("x.html", "x", "story")
    html = site.diagram_html("no-such-diagram", "alt", page.url, "flowchart LR\n A --> B")
    assert "<pre" in html and "flowchart LR" in html and "cdn" not in html
    strict = bs.Site(tmp_path / "t", strict=True)
    with pytest.raises(bs.BuildError):
        strict.diagram_html("no-such-diagram", "alt", page.url, "x")


def test_architecture_mermaid_blocks_embedded_in_library(pages):
    html = pages["library/architecture.html"]
    assert html.count('class="diagram-img diagram-light"') == 5
    assert "mermaid.min.js" not in html


# --- tooltips and glossary -----------------------------------------------------------------------


def test_every_term_exists_and_every_entry_is_used(built, pages):
    entries = {e["term"] for e in built.glossary.values()}
    used = set()
    for rel, html in pages.items():
        for term in re.findall(r' data-term="([^"]+)"', html):
            assert term in entries, (rel, term)
            used.add(term)
    unused = entries - used
    assert not unused, unused


def test_tooltip_markup_is_accessible(pages, built):
    html = pages["index.html"]
    m = re.search(r'<span class="term" tabindex="0" data-term="[^"]+" aria-describedby="(tip-\d+)">.*?<span class="tip-body" role="tooltip" id="\1">', html, re.DOTALL)
    assert m
    css = (built.out / "site.css").read_text(encoding="utf-8")
    assert ":focus-within" in css and ":hover" in css


def test_term_inside_link_or_code_is_refused(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=True)
    page = bs.Page("x.html", "x", "story")
    with pytest.raises(bs.BuildError):
        site.expand('<a href="y.html">{{term:minDCF}}</a>', page)
    with pytest.raises(bs.BuildError):
        site.expand("<code>{{term:minDCF}}</code>", page)
    assert 'class="term"' in site.expand("{{term:minDCF}}", page)


def test_unknown_term_and_unexpanded_tag_fail(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=True)
    page = bs.Page("x.html", "x", "story")
    with pytest.raises(bs.BuildError):
        site.expand("{{term:no-such-term}}", page)
    with pytest.raises(bs.BuildError):
        site.expand("{{bogus:x}}", page)


def test_library_pages_are_never_expanded(pages):
    plan = pages["library/reports-2026-09-26-docs-site-plan.html"]
    assert "{{num:" in plan and '<span class="num"' not in plan and "data-term=" not in plan


# --- search --------------------------------------------------------------------------------------


def test_search_index_lists_every_page_and_is_a_classic_script(built):
    js = (built.out / "search-index.js").read_text(encoding="utf-8")
    assert js.startswith("window.HEARSAY_INDEX=") and "</script" not in js
    data = json.loads(js[len("window.HEARSAY_INDEX="):].strip().rstrip(";"))
    urls = {r["u"] for r in data["pages"]}
    assert urls == {p.url for p in built.pages}
    assert "mindcf" in data["terms"] and "m5" in data["terms"]
    assert all(isinstance(v, list) for v in data["terms"].values())


def test_search_index_loaded_lazily(pages, built):
    assert 'src="search-index.js"' not in pages["index.html"]
    js = (built.out / "site.js").read_text(encoding="utf-8")
    assert "search-index" not in js or "data-search-index" in pages["index.html"]


def test_exports_indexed_by_title_only(built):
    js = (built.out / "search-index.js").read_text(encoding="utf-8")
    assert "iVBORw0KGgo" not in js
    assert len(js.encode("utf-8")) < 1_200_000


# --- citations -----------------------------------------------------------------------------------


def test_citations_all_resolve(built):
    problems = built.check_citations()
    assert not problems, problems[:10]


def test_wrong_number_is_caught(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=True)
    page = bs.Page("x.html", "x", "story")
    page.body = site.expand("holdout {{num:0.0066|README.md#results-at-a-glance}}", page)
    site.pages = [page]
    problems = site.check_citations()
    assert any("0.0066" in p for p in problems)


def test_uncited_number_is_caught(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=True)
    page = bs.Page("x.html", "x", "story")
    page.body = "<p>the holdout is 0.0065 and 1,671 files and 27.4% and 500 rows but 12 layers and 2026-09-26 at 12:03 are fine</p>"
    site.pages = [page]
    problems = site.check_citations()
    flagged = {re.search(r"uncited number '([^']+)'", p).group(1) for p in problems}
    assert flagged == {"0.0065", "1,671", "27.4%", "500"}


def test_missing_source_and_bad_section_fail(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=True)
    page = bs.Page("x.html", "x", "story")
    with pytest.raises(bs.BuildError):
        site.expand("{{num:1|docs/nope.md#x}}", page)
    page.body = site.expand("{{num:0.0065|README.md#no-such-section}}", page)
    site.pages = [page]
    assert any("not found" in p for p in site.check_citations())


def test_code_citation_window(bs, tmp_path):
    site = bs.Site(tmp_path / "s", strict=True)
    page = bs.Page("x.html", "x", "story")
    page.body = site.expand("{{num:0.3|src/hearsay/metrics.py#L31}} {{num:0.3|src/hearsay/metrics.py#L200}}", page)
    site.pages = [page]
    problems = site.check_citations()
    assert len(problems) == 1 and "L200" in problems[0]


def test_num_source_links_point_at_library_pages_offline(pages):
    html = pages["index.html"]
    assert 'data-src="README.md#results-at-a-glance"' in html
    assert re.search(r'data-src="README.md#results-at-a-glance">0\.0065</span><a class="src" href="library/readme.html#results-at-a-glance"', html)


def test_spectra_numbers_carry_the_caveat(pages):
    for rel in ("index.html", "detectors.html"):
        html = pages[rel]
        assert "possibly in-sample" in html
        assert 'data-src="README.md#numbers">0.065</span>' in html


def test_home_page_numbers_present_and_cited(pages):
    html = pages["index.html"]
    expected = {
        "0.135": "README.md#results-at-a-glance", "0.0065": "README.md#results-at-a-glance",
        "0.228": "README.md#results-at-a-glance", "0.239": "README.md#numbers",
        "0.140": "README.md#results-at-a-glance", "0.014": "README.md#results-at-a-glance",
        "0.260": "README.md#results-at-a-glance", "0.267": "README.md#numbers",
        "0.301": "README.md#results-at-a-glance", "0.072": "README.md#results-at-a-glance",
        "0.343": "README.md#results-at-a-glance", "0.065": "README.md#numbers",
        "27.4%": "README.md#numbers", "27.5%": "README.md#orchestration-what-routing-changes",
        "1,671": "README.md#numbers", "0 of 1,671": "README.md#the-eight-forensic-techniques",
        "0.0200": "README.md#orchestration-what-routing-changes", "0.229": "README.md#orchestration-what-routing-changes",
        "0.274": "README.md#orchestration-what-routing-changes",
    }
    for value, src in expected.items():
        assert f'<span class="num" data-src="{src}">{value}</span>' in html, (value, src)
    assert 'class="stat pending"' in html


@STALE
def test_home_page_numbers_against_live_sources(bs, built):
    problems = [p for p in built.check_citations() if p.startswith("index.html")]
    assert not problems, problems


# --- cost element --------------------------------------------------------------------------------


def test_cost_constants_match_metrics_module(built):
    sys.path.insert(0, str(REPO / "src"))
    from hearsay import metrics

    c = built.constants
    assert c["C_FA"] == metrics.C_FA and c["C_MISS"] == metrics.C_MISS and c["PI_SYNTH"] == metrics.PI_SYNTH
    assert abs(c["W_FA"] - 9.3333) < 1e-3 and c["W_MISS"] == 1.0


def test_cost_panel_shows_constants_and_weight(pages):
    for rel in ("index.html", "metric.html"):
        html = pages[rel]
        assert 'class="cost-w">9.33</span>' in html and 'data-src="src/hearsay/metrics.py#L29"' in html
        assert "9.33 missed fakes" in html


def test_cost_expression_matches_cost_at(built, pages):
    sys.path.insert(0, str(REPO / "src"))
    from hearsay import metrics

    expr = re.search(r'data-expr="([^"]+)"', pages["metric.html"]).group(1)
    assert "pfa" in expr and "pmiss" in expr
    for k in (0, 1, 5, 10, 25, 50):
        for m in (0, 2, 20, 80):
            pfa, pmiss = k / 100, m / 100
            val = eval(expr, {"__builtins__": {}}, {"pfa": pfa, "pmiss": pmiss})
            y = np.array([0] * 100 + [1] * 100)
            s = np.zeros(200)
            s[:k] = 1.0  # k real files above the threshold
            s[100:200 - m] = 1.0  # all but m synthetic files above it
            assert abs(val - metrics.cost_at(y, s, 0.5)) < 1e-9, (k, m)


def test_metric_page_states_both_readings(pages):
    html = pages["metric.html"]
    assert "P_FA + 4·P_miss" in html and "higher score as bona fide" in html
    assert 'data-src="README.md#orchestration-what-routing-changes">80%</span>' in html
    assert 'data-src="README.md#orchestration-what-routing-changes">9.7%</span>' in html


# --- hygiene -------------------------------------------------------------------------------------


def test_committed_site_paths_not_gitignored():
    if not (SITE / "index.html").exists():
        pytest.skip("site not built into docs/site")
    files = [str(p.relative_to(REPO)) for p in SITE.rglob("*") if p.is_file()]
    res = subprocess.run(["git", "check-ignore", "--stdin"], cwd=REPO, input="\n".join(files),
                         capture_output=True, text=True, check=False)
    assert res.stdout.strip() == "", res.stdout[:500]


def test_no_audio_under_site():
    assert not [p for p in SITE.rglob("*") if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}]


def test_handoff_link_check_amended_regex(built):
    bad = []
    for p in built.generated_paths(built.out):
        if p.suffix != ".html":
            continue
        scan = re.sub(r"<(pre|code)\b.*?</\1>", "", p.read_text(encoding="utf-8"), flags=re.DOTALL)
        for link in re.findall(r'href="(?!https?:|mailto:|data:)([^"#]+)"', scan):
            if not (p.parent / link).exists():
                bad.append((p.name, link))
    assert not bad, bad[:10]
