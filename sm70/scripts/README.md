# sm70 run scripts (durability copy)

Copied verbatim from `/root/run/*.sh` on 2026-10-24 so the baseline survives
outside a single disk. These define the reproducible baseline: service start
(`run_sm70_serve_ipc.sh`), readiness (`serve_wait.sh`), memory guard
(`ram_guard.sh`), measurement (`measure.sh`, `tp4_prof.sh`), and the
AllReduce ring experiments.

Deliberately NOT included: `/data/src/sm70_env.sh` (holds `SM70_API_KEY`).
Scripts source it at runtime; provide your own copy with the key.

Paths inside the scripts are absolute (`/root/run/...`, `/data/...`) and assume
this machine's layout; they are a record of the baseline, not a portable
installer.
