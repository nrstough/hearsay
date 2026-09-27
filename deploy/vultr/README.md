# Vultr demo host (Sat Sep 26, 2026)

One Ubuntu 24.04 Cloud Compute box (4 vCPU / 8 GB, CPU only) that serves the demo app for
Hrushi's video: nginx on :80 → Next.js on 127.0.0.1:3000 → the pipeline API on 127.0.0.1:8000.
The model, the TSV and the UI are unchanged; the box is a copy and is destroyed after the video.
Record: `docs/reports/2026-09-26_vultr-hosting.md`; spend: `docs/reports/cloud-expense-ledger.md`.

| File | Runs where | Does |
|---|---|---|
| `bootstrap.sh` | box, as root, once | packages, Node 22, 4 GB swap, user `hearsay`, uv, clone, `venv.sh`, `npm ci && npm run build`, units, nginx, ufw (22 and 80 only) |
| `venv.sh` | box, as `hearsay` | the Dockerfile's CPU-torch recipe: the lockfile resolves torch to the CUDA build on Linux, so torch/torchaudio come from the PyTorch CPU index; lightgbm (training only) is skipped |
| `sync_assets.sh <ip>` | Mac | rsyncs the gitignored weights, model bundles, per-file results and the submitted TSV (about 3.3 GB) |
| `hearsay-api.service` | box | uvicorn `hearsay.api:app` on 127.0.0.1:8000, Hub offline, 4 threads |
| `hearsay-warm.service` + `warm.sh` | box | after the API starts, scores one demo clip so the models are resident before the first take |
| `hearsay-web.service` | box | `next start` on 127.0.0.1:3000 against the API |
| `nginx-hearsay.conf` | box | :80 → :3000, 100 MB uploads, 300 s read timeout; nothing routes to :8000 |
| `smoke.sh` | box | health (fusion_v2, four scorers), the three demo uploads against the Mac's numbers, the app through nginx, the TSV export hash |

Order, from the Mac:

```bash
scp -r deploy/vultr root@<ip>:/root/vultr && ssh root@<ip> 'bash /root/vultr/bootstrap.sh'
bash deploy/vultr/sync_assets.sh <ip>
ssh root@<ip> 'systemctl start hearsay-api hearsay-web && sleep 90 && su - hearsay -c "cd hearsay && bash deploy/vultr/smoke.sh"'
```

Redeploy a UI change: `ssh hearsay@<ip> 'cd hearsay && git pull --ff-only && npm run build' && ssh root@<ip> systemctl restart hearsay-web`.
Teardown: destroy the instance in the Vultr console, then add the spend row to the ledger.
