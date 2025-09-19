# PRD — SWE-agent–Based SWE-Bench Evaluation

**Audience:** Coding agent implementing the system.
**Scope:** Exact requirements, constraints, acceptance criteria, and open questions.

## 1. Problem Statement

Build a fast, reliable pipeline to evaluate a catalog of coding models on a curated subset of SWE-bench tasks. The runtime must expose rich repository context with robust editing via SWE-agent, and scoring must use the official SWE-bench evaluator to maintain comparability.

## 2. Goals & Non-goals

**Goals**

- Provide rapid, small-slice iterations that reveal meaningful performance differences.
- Use SWE-agent for repo-aware editing, observability, and batching.
- Score exclusively with the official SWE-bench evaluator.
- Produce a public-quality leaderboard with clear methodology.

**Non-goals (until i4)**

- Full-corpus SWE-bench runs.
- Custom evaluator logic.
- Perfect token/cost accounting (deferred to i4).

## 3. Users & Usage

- **Primary users:** Internal researchers and engineers comparing models and agents.
- **Usage pattern:** Run predefined batches on pinned instance sets, inspect trajectories, and compare leaderboard metrics.

## 4. Functional Requirements

### Execution Runtime

- SWE-agent must run via CLI and support batch execution over SWE-bench instances.
- Emit per-instance trajectory artifacts (tool calls, file edits, logs).
- Allow configuring attempts (`K`), timeouts, worker counts, and model endpoints.

### Dataset & Selection

- i0: Pull the latest SWE-bench Lite snapshot to unblock bootstrap speed, then immediately record the exact dataset fingerprint (HF repo revision/SHA plus retrieval date) in `config/dataset.lock`.
- i1 onward: Treat the revision in `config/dataset.lock` as immutable unless a deliberate update is approved; all dataset downloads must reference that lock file.
- Use SWE-bench Lite for i1 and i2; expand to the locked revision’s broader splits in i3.
- Provide pinned instance lists (`config/instances_i1.yaml`, `config/instances_i3.yaml`) with exact `instance_id`s.
- Record dataset revisions within each run manifest as a cross-check against the lock file.
- Track run wall-clock durations in the manifest; aim for i1 and i2 sweeps to finish within about 45 minutes on the target hardware.

### Scoring

- Use the official SWE-bench evaluator without modification.
- Accept a predictions file and output canonical metrics (resolution rate, etc.).
- Store evaluator outputs under `runs/<run_id>/` and treat them as immutable.

### Model Catalog

The catalog must match the user’s list.

- **i1 (Chutes only):** `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `moonshotai/Kimi-K2-Instruct-0905`, `moonshotai/Kimi-K2-Instruct-75k`, `zai-org/GLM-4.5-FP8`, `zai-org/GLM-4.5-Air`.
- **i2+ additions:** `moonshotai/Kimi-Dev-72B`, `deepseek-ai/DeepSeek-V3.1`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1-0528`, `unsloth/gemma-3-12b-it`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `chutesai/Mistral-Small-3.2-24B-Instruct-2506`, `Qwen/Qwen3-235B-A22B-Thinking-2507`, plus OpenRouter models `openai/gpt-5-mini`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`.

**Availability policy:** If a model is missing, renamed, or unavailable, skip it and report; the batch must continue.

### Budgets & Concurrency

- Attempts: default `K=2`; allow `K=3` per instance with evidence.
- Timeouts: begin at 60 seconds per attempt; raise to 90 seconds or higher only when telemetry confirms insufficiency (i3 may use 180–300 seconds for select tasks).
- Workers: predictions/jobs up to eight (raise to 12 if stable); evaluator workers start at four (raise to six or eight later).
- CPU cap: sustained ≤ 80% of 16 threads.
- Provider quotas: Chutes allows 2k/5k free API requests daily—handle quota exhaustion gracefully and retry later; OpenRouter runs must respect the ~$5 credit by capping max tokens and preferring lower-cost models.

### Artifacts & Data

- Trajectories (per instance), run manifest (versions, seeds, dataset revision from `config/dataset.lock`, model parameters).
- Predictions (`.jsonl`), evaluator outputs, leaderboard (`.csv` plus Markdown summary).
- Store all artifacts under `runs/<run_id>/` and `predictions/`.

### Observability

- Capture timestamps, durations, tool calls, exit codes, and failure reasons for each attempt.
- Ensure any leaderboard row traces back to a `run_id` and supporting artifacts.

### Security

- No outbound network from inside task sandboxes except what SWE-agent needs for model APIs.
- Store secrets in `secrets/credentials.txt` (gitignored) and load at runtime.

### Error Handling

- Partial failures must not abort the batch.
- Standard statuses: `ok`, `provider_failed`, `timeout`, `eval_error`, `skipped_unavailable_model`.

### Token/Cost/Latency (i4)

- Add input/output/total tokens, USD cost, P50/P90/P99 latency.
- Validate token/cost reporting against provider dashboards on sampled runs.

## 5. Model Integration Details

- Prefer OpenAI-compatible HTTP endpoints; document any required proxy or adapter settings (`api_base`, model name, headers) in `config/sweagent.yaml`.
- Set per-model parameters: temperature, max output tokens, seed (for determinism), context window.
- Implement backoff and retry for HTTP 429/5xx; after `N` failed attempts, skip and report the model.

## 6. Metrics & Leaderboard

- **v1 (i2):** pass@1, pass@K, wall time per model, attempts used, run-fail rate.
- **v2 (i3):** v1 metrics plus per-repo breakdown and difficulty grouping.
- **v3 (i4):** v2 metrics plus tokens (in/out/total), USD cost, latency P50/P90/P99, cost-per-resolved-instance.

## 7. Acceptance Criteria

- **i0:** Working SWE-agent batch on two to three Lite/dev instances; artifacts present; CPU < 80%; `config/dataset.lock` populated with the dataset fingerprint.
- **i1:** Pinned six to eight instances; six Chutes models run; best model significantly beats no-patch baseline; trajectories readable; evaluator outputs clean and reproducible; dataset revision matches the lock; end-to-end sweep completes within the 45-minute target.
- **i2:** Multi-attempt enabled; additional models integrated; leaderboard v1 produced.
- **i3:** 15–20 total tasks (easy plus multi-patch plus NumPy/Pandas); leaderboard v2 produced.
- **i4:** Token/cost/latency validated against samples; leaderboard v3 produced.

## 8. Constraints & Policies

- Pin instance IDs, dataset revision (`config/dataset.lock`), seeds/temperatures, SWE-agent plus evaluator versions.
- Treat `runs/<run_id>/` as immutable; never overwrite.
- Increase timeouts strictly based on profiling or telemetry evidence.
- Skip and report unavailable models; never block the batch on a single provider.

## 9. Open Questions

- When we intentionally bump the dataset revision, what validation steps (diffing tasks, rerunning baselines) are required before accepting a new lock?
- Do we publish the final leaderboard outside the repo (for example, Gist or static site)?
- Are there per-model temperature or seed overrides beyond defaults?

## 10. Deliverables Checklist

- `config/sweagent.yaml` containing model list, attempts, timeouts, workers, and instance data source.
- `config/dataset.lock` storing HF revision SHA and retrieval date.
- `config/instances_i1.yaml` (six to eight Lite/dev IDs) and `config/instances_i3.yaml` (15–20 expanded IDs).
- `secrets/credentials.txt` template plus `tools/env.sh` loader (gitignored).
- Repeatable run script or CLI guide that executes i1 end-to-end and writes artifacts to `runs/<run_id>/`.
- Predictions `.jsonl` files per model/batch.
- Evaluator outputs stored per `run_id`.
- `leaderboard.csv` with accompanying Markdown summary per iteration.
- Short validation memo for each iteration confirming exit gates.
