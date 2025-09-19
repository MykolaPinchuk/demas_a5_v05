# Iteration i0 — Validation Notes

## Scope

Bootstrap the DEMAS pipeline by running a minimal SWE-agent workflow across a
three-instance SWE-bench Lite smoke slice. Emphasis: end-to-end operability,
dataset fingerprinting, and artefact structure. Performance metrics and
multi-model comparisons arrive in later iterations.

## Checklist

- [x] Dataset cache present at `data/swe_bench_lite/dev` (or alternative
      offline path `poc/data/swe_bench_lite/dev`).
- [x] `config/dataset.lock` populated with `dataset`, `revision`, `split`, and
      `retrieved_at`.
- [x] `runs/<run_id>/manifest.json` lists each instance with status, attempts,
      timestamps, and failure reason when relevant.
- [x] `predictions.jsonl` emitted alongside the manifest.
- [x] At least one instance status `ok` with corresponding `success.json`
      (`i0_20250919_032854` has three successes).
- [x] Partial failures leave artefacts intact for debugging (`i0_20250919_033335`
      captures `llm_failed` for `pydicom__pydicom-1694`).

## Summary & Artefacts

- Successful full-network sweep: `runs/i0_20250919_032854/` (Chutes +
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`, 3/3 solved).
- Weak-model exercise with controlled failure: `runs/i0_20250919_033335/`
  (`unsloth/gemma-3-12b-it`, showcases `llm_failed`).
- Offline bootstrap reference: `runs/i0_20250919_032525/` (uses repo cache and
  deterministic replacements).

Hand off these run IDs and the dataset fingerprint in `config/dataset.lock` to
the next iteration.

## Open Questions

1. **Hard-coded overrides vs. policies:** For i1 keep literal replacements for
   the smoke slice, but start a backlog item to replace them with a configurable
   ``instance_overrides`` section in `config/sweagent.yaml` once we introduce
   more tasks.
2. **Dataset fingerprint changes:** Treat any new fingerprint as a breaking
   change—rerun the i0 slice and refresh the baseline manifests before
   promoting the lock. Document the delta in a short memo stored alongside the
   lock file.
3. **Resource observability:** Defer automated CPU tracking to i1; in the
   interim, capture `time` command output for the run scripts to produce wall
   time and rough CPU usage snapshots.
4. **Telemetry schema:** Keep the current manifest structure for i1 and expand
   it incrementally with evaluator metrics (pass@1, attempts used). Agree on a
   JSON schema before i2 to avoid rework.
5. **Containerisation:** Evaluate whether switching to a Docker-based runtime in
   i1 improves reproducibility and isolation; weigh it against added build time
   and networking constraints for model APIs.

Capture evidence for each checklist item in commit messages or run logs before
progressing to i1.
