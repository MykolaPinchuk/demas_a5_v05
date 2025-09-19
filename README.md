# DEMAS SWE-Bench Evaluation

This repository houses the greenfield implementation of the DEMAS SWE-agent
benchmarking pipeline. Iteration **i0** focuses on a tiny SWE-bench Lite smoke
slice to validate dependencies, dataset handling, and run-time artefact layout
before scaling to broader iterations described in `docs/`.

## Repository Layout

- `docs/` — Product requirements, long-range plan, and validation notes.
- `src/demas/` — Python package powering the CLI and agent orchestration.
- `config/` — Dataset/instance configuration files (mutable per iteration).
- `tools/` — Reserved for helper scripts (will expand in later iterations).
- `secrets/credentials.txt` — API keys for model providers (gitignored).

## Getting Started (i0)

1. **Python environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

   If pip cannot reach package indexes, you can operate with
   `PYTHONPATH=src` instead of installing the package—dependencies used in i0
   (`datasets`, `pyyaml`, `requests`) are already present in the reference
   environment.

2. **Provider credentials**

   Populate `secrets/credentials.txt` with key/value pairs required by your
   providers. At minimum one of the following keys is expected by the i0 agent:

   ```
   CHUTES_API_KEY=...
   CHUTES_API_KEY_ALT=...
   OPENROUTER_API_KEY=...
   ```

3. **Download the dataset**

   The CLI fetches SWE-bench Lite on demand. If you are operating in an
   air-gapped environment, point the command to a pre-existing dataset cache
   (for example, the POC bundle ships one under `poc/data/swe_bench_lite/dev`):

   ```bash
   PYTHONPATH=src python -m demas.cli \
     --model instant_empty_submit \
     --provider offline \
     --dataset-dir poc/data/swe_bench_lite/dev
   ```

   The command ensures the dataset is cached locally, records the fingerprint in
   `config/dataset.lock`, and, when `--provider offline` is used, applies the
   deterministic string replacements for the i0 slice without contacting an
   external LLM.

4. **Execute the smoke slice**

   With credentials and network access, swap in a real provider/model pair.
   When running offline, keep `--provider offline` and rely on the embedded
   replacements (two of the three tasks will pass without an LLM):

   ```bash
   PYTHONPATH=src python -m demas.cli \
     --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
     --provider chutes \
     --dataset-dir data/swe_bench_lite/dev
   ```

   Each execution creates a timestamped folder under `runs/` with per-instance
   subdirectories containing prompts, patches or manual replacement markers,
   command logs, and success/failure summaries. A run-level `manifest.json` and
   `predictions.jsonl` capture the sweep output.

## Validation Checklist (i0)

- Dataset fingerprint captured in `config/dataset.lock` after the first run.
- `runs/<run_id>/<instance_id>/` contains prompts, LLM responses, patches, and
  pytest logs.
- At least one instance reaches status `ok` with `success.json` documenting the
  passing attempt.
- CPU usage stays comfortably under the 80% target (verify externally).

## Next Steps

- Flesh out `tools/` with helper scripts (dataset lock updater, run wrappers).
- Expand configuration coverage for i1 pinned instances and additional models.
- Integrate the official SWE-bench evaluator and capture metrics in
  `runs/<run_id>/`.
