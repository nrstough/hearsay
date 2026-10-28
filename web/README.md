# web/: superseded static prototype ("Lumenix"), mock data only

This folder is an early static mock-up of the HEARSAY interface, made on Saturday Sep 26 before
the Next.js app was wired to the pipeline (the frontend handoffs of ~09:20 and ~13:10 bracket it).
It is kept for the record. **It is not the product and none of its numbers come from HEARSAY.**

- It never calls the pipeline or the API: there is no `fetch` anywhere in `app.js`.
- The batch table (`populateBatchQueue` in `app.js`) is hard-coded rows with made-up file names,
  verdicts and confidences.
- The TSV download (`triggerTsvDownload` in `app.js`) writes 1,671 rows of *random* scores. Never
  submit or compare against a file it produced.
- `serve.py` is a throwaway static server for this folder, run by hand. Nothing deployed serves
  or links to it: the demo host's nginx proxies only to the Next.js app
  (`deploy/vultr/nginx-hearsay.conf`), the docs site and the top-level README do not link here,
  and the Docker image excludes `web` (`.dockerignore`).

The product:

- **UI:** the Next.js app under `src/app/` and `src/components/` (`npm run dev`), which reads the
  runner's per-file JSON and calls the API.
- **API:** `src/hearsay/api.py` (FastAPI over `hearsay.pipeline.analyze_clip`: `/analyze`,
  `/results`, `/health`); contract in `docs/handoffs/2026-09-26_frontend-contract.md`.
- **Scoring a test set:** `scripts/run_pipeline.py` (see the top-level `README.md`, "Reproduce it").
