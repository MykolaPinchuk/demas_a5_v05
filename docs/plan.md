# Implementation Plan — Greenfield SWE-Bench Evaluation with SWE-agent (i0→i4)

**Audience:** Coding agent who will build and operate this project.
**Scope:** End-to-end plan from first bootstrap to a validated leaderboard using SWE-agent for execution and the official SWE-bench evaluator for scoring.
**Principle:** Prioritize rapid iteration—start small and fast, collect evidence, only then increase budgets.

## 0. Objectives & Guardrails

**Primary goal:** Produce a trustworthy leaderboard of coding models on a curated subset of SWE-bench using SWE-agent for repo-aware edits and the official SWE-bench evaluator for scoring. 

- Maintain fast feedback loops: tiny task slices, short timeouts, conservative concurrency; scale only after clearing the exit gates.
- Respect compute budget: target at most 80% of local CPU (16 threads → plan around 12 workers max); keep full-suite runs under about 45 minutes during i1–i2.
- Stay within provider budgets: Chutes offers 2k/5k free API calls per day (expect retries to back off once exhausted); OpenRouter usage should stay within the $5 credit by preferring cheaper models and tight max-tokens.
- Handle secrets via `secrets/credentials.txt` (gitignored) and load keys at runtime.
- Use the exact model catalog provided in PRD §5.

## 1. Iterations

### i0 — Bootstrap (same day)

**Outcome:** A fresh repo runs a single SWE-agent batch on two to three Lite/dev instances with one model and produces artifacts.

Do not overengineer in i0. Get to running an agent as soon as possible. Then debug issues which will likely arise. To help with i0, you can consult with codebase in poc. poc is a bare minimum throwaway codebase which implements swe-agent and runs in successfully on three tasks. To be clear, this codebase was build as a throwaway project and is not maintainable. You can copy blocks of code from there, but be very careful. I include poc mostly for reference and to de-risk i0.

1. Create a clean repo and Python environment.
2. Install SWE-agent and its minimal dependencies.
3. Confirm the `sweagent` CLI runs locally and emits outputs.
4. Prepare `secrets/credentials.txt` (gitignored) and load keys via environment variables at runtime.
5. Pull the latest SWE-bench Lite snapshot, run the tiny slice, and immediately record the dataset fingerprint (HF repo revision plus retrieval date) in `config/dataset.lock` for future runs.
6. Pin the smoke-test instance IDs in `config/instances_i0.yaml` (currently three Lite/dev SQLFluff tasks) so tooling can derive filters programmatically.
7. Verify the end-to-end bootstrap using the `instant_empty_submit` model before burning real API budget; swap in Chutes/OpenRouter identifiers once credentials are confirmed.
- No dataset lock yet. Just using the latest snapshot is enough for i0.

**Exit gate:**

- CLI completes a tiny slice (two to three instances) and writes trajectory artifacts.
- Agent solves at least one task.
- CPU stays under 80%, with no crashes.
- `config/dataset.lock` captures the dataset revision used.
- Do not build anything for i1 until i0 is fully complete.

### i1 — Working Slice (weekend-scale)

**Outcome:** Reproducible results on six to eight very easy Lite/dev instances across six models (Chutes) with observable trajectories and a clear spread versus the no-patch baseline.

- **Dataset:** Use the revision recorded in `config/dataset.lock`; treat it as immutable unless deliberately updated.
- **Instances:** Use SWE-bench Lite, dev split; the pinned working slice lives in `config/instances_i1.yaml` (currently eight SQLFluff/Marshmallow/PVLib/PyDicom tasks). Update only with a documented rationale.
- **Automation:** `tools/run_i1.sh` drives the Chutes model set across the i1 slice by chaining `bootstrap_i0.sh`; use `--execute` once credentials are configured. `tools/run_model_batch.py` is the provider-agnostic launcher backed by `config/model_catalog.yaml` for Chutes plus OpenRouter runs.
- **Reporting:** `tools/collect_manifests.py` converts `runs/*/manifest.json` into CSV summaries; extend it once i1 metrics (pass@1, etc.) are available.
- **Leaderboard:** `tools/generate_leaderboard.py` joins evaluator reports to compute resolution counts/rates, producing `runs/leaderboard.csv` as an interim scorecard until full i2 metrics land.
- **Models (Chutes only):** `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `moonshotai/Kimi-K2-Instruct-0905`, `moonshotai/Kimi-K2-Instruct-75k`, `zai-org/GLM-4.5-FP8`, `zai-org/GLM-4.5-Air`.
- **Policy:** If a model is unavailable or renamed, skip and report.
- **Budgets:** Attempt count `K=2` per instance; start with 45 seconds per attempt and raise (for example, to 60 seconds) only when telemetry shows it is required for specific tasks.
- **Concurrency:** Predictions/jobs up to eight; evaluator workers start at four.
- **Artifacts:** Per-instance trajectories, per-run manifest, predictions file(s).

**Exit gates:**

- Best model has a pass rate at least 50% higher than the no-patch baseline on the pinned set.
- Malformed-patch and path-not-found error classes remain rare (SWE-agent handles editing).
- Re-running the batch on the same slice reproduces outcomes within normal variance.
- Achieve all i1 requirements before advancing; obtain human validation that results are solid.

### i2 — Scale Models + Multi-attempt + Leaderboard v1

**Outcome:** Broader model coverage (Chutes plus OpenRouter), multi-attempt enabled, leaderboard with pass@1/pass@K and operational metrics.

- Continue using the dataset revision locked in `config/dataset.lock` unless a deliberate update is executed (which must regenerate the lock file).
- Add multi-attempt policy (`K=2` default; allow `K=3` where justified).
- Expand models per PRD §5 (Chutes plus OpenRouter).
- Enable targeted test runs (fast checks) in SWE-agent config where appropriate; keep per-model wall-clock under roughly 60 minutes even as coverage expands.
- Produce `leaderboard.csv` v1 with pass@1, pass@K, wall time per model, attempts used, run-fail rate.
- Consider raising evaluator workers to six to eight if CPU usage is under 80% and I/O remains stable.

**Exit gates:**

- At least eight to ten models produce results.
- Leaderboard shows a non-trivial spread; reruns reproduce results.

### i3 — Broader Difficulty (Multi-patch + NumPy/Pandas)

**Outcome:** Total of 15–20 tasks mixing easy and medium difficulty; SWE-agent retains shell-like editing defaults; longer timeouts applied only when proven necessary.

- Keep the six to eight easy items; add six to eight medium multi-patch tasks; add three to four NumPy/Pandas-focused tasks, all pulled from the locked dataset revision.
- Increase attempt timeout per instance (up to 120–180 seconds) only when evidence shows the 90-second budget is insufficient.
- Publish leaderboard v2: v1 metrics plus per-repo breakdown.

### i4 — Token & Cost Accounting + Latency

**Outcome:** Standardized token counts, USD cost, and latency (P50/P90/P99) added to the leaderboard.

- Collect provider-reported token data; fall back to official tokenizer, then documented approximation.
- Compute USD cost as `(prompt $/1K * input tokens + completion $/1K * output tokens)`.
- Validate token/cost metrics on a small sample against provider dashboards.
- Release leaderboard v3 (v2 metrics plus tokens, cost, latency).

## 2. Target Repo Layout (No Code Yet)

```
/
├── docs/
│   ├── plan.md                  # this file
│   └── prd.md                   # the PRD
├── config/
│   ├── sweagent.yaml            # SWE-agent runtime config
│   ├── instances_i0.yaml        # smoke-test instances for i0
│   ├── instances_i1.yaml        # pinned instance_ids for i1
│   ├── instances_i3.yaml        # expanded set for i3
│   ├── model_catalog.yaml       # provider/model overrides (Chutes + OpenRouter)
│   ├── leaderboard_columns.yaml # metrics emitted per iteration
│   └── dataset.lock             # dataset revision fingerprint
├── runs/                        # outputs per run_id (immutable)
├── predictions/                 # predictions files per model/batch
├── secrets/
│   └── credentials.txt          # gitignored API keys (KEY=...)
├── tools/
│   └── env.sh                   # helper to export keys from credentials.txt
└── README.md                    # runbook for operators
```

## 3. Operational Runbook (Per Iteration)

1. **Prepare environment:** Install SWE-agent; verify CLI works; record versions in the run manifest.
2. **Load secrets:** Source `tools/env.sh` to export keys from `secrets/credentials.txt`.
3. **Lock dataset revision:**
   - i0: fetch the latest SWE-bench Lite snapshot, record its HF repo revision and retrieval date in `config/dataset.lock`.
   - i1+: read `config/dataset.lock` and ensure all dataset downloads use that revision; update the lock only with an explicit change request.
   - Use `tools/update_dataset_lock.py` to stamp new revisions, ensuring the lock file stays structured.
4. **Select instances:** Generate and commit pinned lists (`config/instances_i1.yaml`, `config/instances_i3.yaml`).
5. **Configure models:** Update `config/sweagent.yaml` with model endpoints, timeouts, attempts, workers per iteration.
6. **Run batch:** Execute SWE-agent on the pinned slice; store outputs in `runs/<run_id>/` and log wall-clock duration versus the 45-minute target.
7. **Evaluate:** Run the official SWE-bench evaluator on predictions to compute scores.
   - Invoke `tools/run_evaluation.py` (dry run first) to wrap the evaluator CLI and capture metadata.
8. **Aggregate:** Update `leaderboard.csv` (columns per iteration).
9. **QA gate:** Confirm exit gates before moving ahead.

## 4. Observability & Traceability

- Preserve trajectory artifacts, prompts, tool calls, durations, exit codes.
- Store dataset revision (from `config/dataset.lock`), SWE-agent version, evaluator version.
- Ensure every leaderboard cell maps back to a `run_id` and its artifacts.

## 5. Risks & Mitigations

- **Provider API changes:** Adapter layer logs and skips unavailable models; report clearly.
- **Long setup times:** Start with Lite/dev slices, short timeouts, gradual budget increases.
- **Non-determinism:** Pin instance IDs; fix seeds/temperatures; document variance bands.

## 6. Deliverables by Iteration

- **i0:** Working CLI smoke on two to three instances; run manifest captured; `config/dataset.lock` populated.
- **i1:** Pinned six to eight instances; predictions per model; evaluator outputs; validation memo; leaderboard v0.
- **i2:** Multi-attempt enabled; expanded model set; leaderboard v1.
- **i3:** 15–20 tasks plus per-repo breakdown; leaderboard v2.
- **i4:** Token/cost/latency validated; leaderboard v3.

## 7. Miscellaneous

Some agents found the following helpful snippet: `CHUTES_BASE_URL = os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")`.

Always use timeouts when running an agent.
