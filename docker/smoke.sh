#!/bin/bash
# Image smoke test (run spec T6): three files (WAV, MP3, FLAC) and a reversed template through
# the image with --network none; the TSV must have three rows in template order; a second run
# must agree within 1e-6 (x86 multithreaded BLAS is not guaranteed bit-exact; it was identical on
# the recorded builds); a run without the offline variables must be refused; the in-image
# self-checks must pass. Everything lives under
# $PWD/outputs/docker/smoke because Colima bind-mounts only paths under $HOME.
# Usage: bash docker/smoke.sh [image]   (default hearsay:latest)
set -euo pipefail
IMAGE="${1:-hearsay:latest}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$PWD/outputs/docker/smoke"
rm -rf "$ROOT" && mkdir -p "$ROOT/data" "$ROOT/out1" "$ROOT/out2"
RUN="docker run --rm --network none --platform linux/amd64"

echo "== self-checks (no network)"
$RUN --entrypoint python "$IMAGE" -c "import platform, torch, transformers, sklearn, numpy; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$RUN --entrypoint python "$IMAGE" -c "import hearsay.embed as e, hearsay.pipeline as p; assert (e.REPO/'weights'/'wav2vec2-xls-r-300m'/'config.json').exists(), e.REPO; assert p.PROBE_DIR.exists() and p.HC_DIR.exists() and p.CONSTANTS_PATH.exists(), (p.PROBE_DIR, p.HC_DIR, p.CONSTANTS_PATH); print('paths OK under', e.REPO)"
$RUN --entrypoint python "$IMAGE" -c "import numpy as np, librosa; librosa.feature.mfcc(y=np.zeros(16000, 'f4'), sr=16000); print('librosa/numba OK')"
$RUN --entrypoint python "$IMAGE" -c "import sys; assert 'lightgbm' not in sys.modules; import hearsay.pipeline, hearsay.detectors.engineered; assert 'lightgbm' not in sys.modules; print('no lightgbm')"
$RUN --entrypoint python "$IMAGE" docker/assets.py verify --root /app --manifest /app/assets.json --full
$RUN --entrypoint ffmpeg "$IMAGE" -version | head -1
$RUN --entrypoint python "$IMAGE" -c "import numpy as np; from hearsay.detectors.base import ClipContext, safe_run; from hearsay.detectors import speaker_drift as sd; x=(0.1*np.random.default_rng(0).standard_normal(48000)).astype('f4'); r=safe_run(sd.DETECTOR, ClipContext.from_array(x)); assert r.status=='ok', r.error; print('speaker_drift OK (offline ECAPA):', r.evidence[:60])"

echo "== three files + reversed template"
ffmpeg -nostdin -v error -y -f lavfi -i "sine=frequency=220:sample_rate=16000:duration=2" -ac 1 "$ROOT/data/a_sine.wav"
ffmpeg -nostdin -v error -y -f lavfi -i "anoisesrc=color=pink:sample_rate=44100:duration=2:seed=1" -ac 2 -b:a 96k "$ROOT/data/b_noise.mp3"
ffmpeg -nostdin -v error -y -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=1.5" -ac 1 "$ROOT/data/c_tone.flac"
printf 'filename\tcm-score\nc_tone.flac\t0.5\nb_noise.mp3\t0.5\na_sine.wav\t0.5\n' > "$ROOT/template.tsv"

echo "== offline enforcement: a container started without an offline variable must be refused"
mkdir -p "$ROOT/outneg"
if $RUN -e TRANSFORMERS_OFFLINE=0 -e HEARSAY_TEAM=smoke -v "$ROOT/data:/data:ro" -v "$ROOT/template.tsv:/tmpl/key.tsv:ro" \
        -v "$ROOT/outneg:/out" -e HEARSAY_TEMPLATE=/tmpl/key.tsv "$IMAGE" > "$ROOT/neg.log" 2>&1; then
  echo "smoke: the container ran with TRANSFORMERS_OFFLINE=0; --require-offline is not enforced" >&2; exit 1
fi
grep -qi "offline" "$ROOT/neg.log" || { echo "smoke: refusal did not mention the offline variables:" >&2; tail -3 "$ROOT/neg.log" >&2; exit 1; }
if ls "$ROOT/outneg"/*.tsv >/dev/null 2>&1; then echo "smoke: a TSV was written despite the refusal" >&2; exit 1; fi
[ -z "$(ls -A "$ROOT/outneg")" ] || { echo "smoke: the refused run left files in /out: $(ls "$ROOT/outneg")" >&2; exit 1; }
echo "smoke: refused without the offline variables on the real input, /out left empty (as required)"

for i in 1 2; do
  $RUN -v "$ROOT/data:/data:ro" -v "$ROOT/template.tsv:/tmpl/key.tsv:ro" -v "$ROOT/out$i:/out" \
       -e HEARSAY_TEMPLATE=/tmpl/key.tsv -e HEARSAY_TEAM=smoke "$IMAGE"
done
TSV="$ROOT/out1/smoke_predictions.tsv"
[ -f "$TSV" ] || { echo "smoke: no TSV written" >&2; exit 1; }
[ "$(wc -l < "$TSV")" -eq 4 ] || { echo "smoke: expected 4 lines, got $(wc -l < "$TSV")" >&2; exit 1; }
[ "$(head -1 "$TSV")" = $'filename\tcm-score' ] || { echo "smoke: bad header" >&2; exit 1; }
[ "$(cut -f1 "$TSV" | tail -n +2 | tr '\n' ' ')" = "c_tone.flac b_noise.mp3 a_sine.wav " ] || { echo "smoke: row order is not the template's" >&2; exit 1; }
python3 - "$TSV" "$ROOT/out2/smoke_predictions.tsv" <<'PY'
import csv, sys
a, b = (list(csv.reader(open(p), delimiter="\t")) for p in sys.argv[1:3])
assert [r[0] for r in a] == [r[0] for r in b], "smoke: the two runs list different files or orders"
d = max(abs(float(x[1]) - float(y[1])) for x, y in zip(a[1:], b[1:]))
assert d < 1e-6, f"smoke: repeat runs differ by {d} (> 1e-6)"
print(f"smoke: two runs agree (same ids and order, max |diff| {d:.1e}; x86 multithreaded BLAS is not bit-exact)")
PY
python3 - "$ROOT/out1/run_meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("preflight:", json.dumps(m.get("preflight")))
flags = m.get("flags") or {}
decode_errors = m.get("n_decode_error", sum(v for k, v in flags.items() if "decode" in k))
print("gated:", m.get("n_gated"), "decode errors:", decode_errors, "flags:", flags, "threads:", m.get("threads"), "wall:", m.get("wall_seconds"), "s")
assert decode_errors == 0, "a smoke file failed to decode in the image"
PY
echo "== --flip: the pre-flipped twin re-fused from the cache"
$RUN -v "$ROOT/data:/data:ro" -v "$ROOT/template.tsv:/tmpl/key.tsv:ro" -v "$ROOT/out1:/out" \
     -e HEARSAY_TEMPLATE=/tmpl/key.tsv -e HEARSAY_TEAM=smoke "$IMAGE" --flip
FL="$ROOT/out1/smoke_predictions_FLIPPED.tsv"
[ -f "$FL" ] || { echo "smoke: --flip wrote no smoke_predictions_FLIPPED.tsv" >&2; exit 1; }
python3 - "$TSV" "$FL" "$ROOT/out1/run_meta.json" <<'PY'
import csv, itertools, json, sys
a, b = (list(csv.reader(open(p), delimiter="\t")) for p in sys.argv[1:3])
assert a[0] == b[0] == ["filename", "cm-score"] and [r[0] for r in a] == [r[0] for r in b], "flipped file order differs"
pa, pb = [float(r[1]) for r in a[1:]], [float(r[1]) for r in b[1:]]
assert all(0.0 <= v <= 1.0 for v in pa + pb), "scores outside [0, 1]"
# Gated (non-speech) files form a pinned block below 0.001 in both polarities by the runner's
# policy, so only determinate rows (>= 0.001 in both files) must reverse their order.
det = [(x, y) for x, y in zip(pa, pb) if x >= 0.001 and y >= 0.001]
for (x1, y1), (x2, y2) in itertools.combinations(det, 2):
    if x1 != x2:
        assert (x1 < x2) == (y1 > y2), f"flipped ranking is not the reverse: {pa} vs {pb}"
assert all((x < 0.001) == (y < 0.001) for x, y in zip(pa, pb)), "a file is gated in one polarity only"
m = json.load(open(sys.argv[3]))
pol = (m.get("version") or {}).get("polarity") or m.get("polarity")
assert pol == "flipped", f"run_meta polarity is {pol!r}, expected 'flipped'"
print(f"smoke: --flip wrote the FLIPPED twin ({len(pb)} rows, same order, reversed where untied), polarity recorded")
PY
echo "== smoke OK: $TSV"
cat "$TSV"
