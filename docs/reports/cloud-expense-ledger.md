# Cloud expense ledger (HEARSAY)

Every rental appends a row here (`scripts/cloud/launch.sh`), the reaper appends the destroy,
and `scripts/cloud/teardown_check.sh` appends the credit left. Balance: `uvx vastai show user --raw`.

## Payments in

| Date | Vendor | Method | Amount |
|---|---|---|---|
| before Sep 26 | vast.ai | card | credit $13.37 at handoff (01:30) |

## Charges

| Date | Vendor | Instance / item | Role | GPU | Storage | Bandwidth | Total | |
|---|---|---|---|---|---|---|---|---|
| 2026-09-26 04:29 | vast.ai | 52716242 (50894989) | pilot | 0.611111111111111 A100 SXM4 | | | est 1.5 h | 
| 2026-09-26 04:51 | vast.ai | 52716242 | pilot | destroyed (failed: FAIL fold4_nsa_extra) | | | | 
| 2026-09-26 04:54 | vast.ai | 52718897 (50894989) | pilot | 0.611111111111111 A100 SXM4 | | | est 1.0 h | 
| 2026-09-26 05:47 | vast.ai | 52725429 (51341862) | abl_extra | 0.7055555555555555 A100 SXM4 | | | est 1.5 h | 
| 2026-09-26 06:00 | vast.ai | 52726831 (24559409) | abl_cons | 0.811111111111111 A100 PCIE | | | est 1.2 h | 
| 2026-09-26 06:39 | vast.ai | 52726831 | abl_cons | destroyed (failed: FAIL full_frozen) | | | | 
| 2026-09-26 06:55 | vast.ai | 52725429 | abl_extra | destroyed (failed: FAIL full_frozen) | | | | 
| 2026-09-26 07:34 | vast.ai | teardown check | | | | | credit left $33.28 | 
| 2026-09-26 07:35 | vast.ai | M5 total (7 rentals: 2 stalled at boot, pilot x2, ablation x2, conservative x1) | all runs | A100 SXM4/PCIe $0.61–0.81/h | included | included | **$5.09** (credit $38.37 → $33.28) | 
