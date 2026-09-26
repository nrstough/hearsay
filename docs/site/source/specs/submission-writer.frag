<!-- {"title": "The submission writer and the experiment log", "group": "Outputs", "order": 60, "lead": "One function writes the TSV and refuses anything that would make it invalid: a wrong header, a duplicate file, a non-finite or out-of-range score, or an existing path. Every TSV gets a row in a tracked log with its validation score. A submitted file is never overwritten.", "sources": ["src/hearsay/submission.py", "CLAUDE.md"]} -->
<h2 id="plain">In plain words</h2>
<p>The deliverable, the {{term:TSV}}, is a text file with a header and one row per test file, in the sponsor's order. The cheapest way to lose the weekend would be a malformed file, so the writer checks everything it can and reads the file back to confirm it. The log exists so that every number ever submitted can be traced to a file and a validation score; the project rule is that a submitted TSV is never renamed or overwritten.</p>
<h2 id="io">What goes in and comes out</h2>
<p>In: the ordered file ids and their scores. Out: <code>&lt;team&gt;_predictions.tsv</code> with header <code>filename&lt;TAB&gt;cm-score</code>, and a row appended to <code>submissions/log.csv</code>.</p>
<h2 id="where">Where it lives</h2>
<p>{{src:src/hearsay/submission.py|src/hearsay/submission.py}}: <code>score_files</code>, <code>write_submission</code>, <code>append_log</code>, <code>preflight</code>, and the accepted audio extensions.</p>
<h2 id="params">Rules and where each is set</h2>
<ul>
<li><code>write_submission</code> refuses, in order: an existing path, a length mismatch between ids and scores, an empty submission, duplicate ids, non-finite scores, scores outside [0, 1]; then it writes with exclusive-create and reads the file back to assert the header, order and exact values ({{src:src/hearsay/submission.py#L73|submission.py, lines 73–110}}).</li>
<li>The log columns are <code>timestamp, rung, validation_score, validation_score_clean_only, csv_path, notes</code>; the clean-only column defaults to the validation score when no augmentation was used; the path is stored relative to the repository ({{src:src/hearsay/submission.py#L29|submission.py, lines 29–31}} and {{src:src/hearsay/submission.py#L113|lines 113–133}}). The column named <code>csv_path</code> is a historical name for the submission path.</li>
<li><code>score_files</code> never loses a row: a decode error, a scorer exception or a non-finite score yields the fallback score plus a flag ({{src:src/hearsay/submission.py#L44|submission.py, lines 44–70}}).</li>
<li>The library's <code>preflight</code> scores silence and a chord twice and requires exact equality; the runner uses its own tolerant version.</li>
</ul>
<h2 id="status">Status</h2>
<p>Built in the first hour and unchanged. The constant "always real" rollback TSV existed before any model did.</p>
<h2 id="numbers">Numbers</h2>
<p>The log is the source of truth on numbers where documents disagree (a project rule). The shipped TSV's row is timestamped 12:03 on Saturday; the file was sent at 12:30.</p>
<h2 id="limits">Known limits</h2>
<p>The writer validates format, not content: a valid file with a bad ranking is still valid.</p>
<h2 id="tests">Tests that pin it</h2>
<p><code>tests/test_audio_and_submission.py</code>: one finite score per file in order, a scorer exception is survived, the exact TSV bytes, never overwrites, rejects a length mismatch, a duplicate, <code>NaN</code>, <code>1.5</code>, <code>−0.1</code> and an empty submission leaving no file behind, and the log row.</p>
<h2 id="sources">Sources</h2>
<p>{{src:CLAUDE.md#conventions|CLAUDE.md, "Conventions"}}; {{src:README.md#where-each-artifact-lives|README, "Where each artifact lives"}}.</p>
