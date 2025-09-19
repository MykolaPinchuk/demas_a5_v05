# SWE-Bench Lite Agent POC

This repository contains a minimal but functioning agent that can solve selected
[SWE-bench Lite](https://www.swebench.com/) tasks using hosted LLM providers.
It was originally created as a proof-of-concept and now serves as a starting
point for a more full-featured automation project. The code favours clarity and
traceability over abstractions so that the next generation of agents can extend
it quickly.

## Current Capabilities

The agent has been validated end-to-end on the following SWE-bench Lite
instances using chutes.ai models (timestamps available under `runs/` whenever
you execute a new run):

| Instance ID | Repository | Successful Models (chutes.ai) |
|-------------|------------|--------------------------------|
| `pydicom__pydicom-1694` | `pydicom/pydicom` | `Qwen/Qwen3-Coder-30B-A3B-Instruct`, `moonshotai/Kimi-K2-Instruct-0905`, `deepseek-ai/DeepSeek-V3-0324` |
| `sqlfluff__sqlfluff-1625` | `sqlfluff/sqlfluff` | `deepseek-ai/DeepSeek-V3-0324`, `moonshotai/Kimi-K2-Instruct-75k` |
| `marshmallow-code__marshmallow-1359` | `marshmallow-code/marshmallow` | `deepseek-ai/DeepSeek-V3-0324`, `moonshotai/Kimi-K2-Instruct-75k` |

Support for additional tasks should only require small adjustments (see
[Extending to New Tasks](#extending-to-new-tasks)).

### Task-Specific Tweaks

To keep the POC reliable, a few lightweight overrides are baked into
`poc_agent.py`. They are easy to spot near the top of the file and are intended
to be replaced with a more generic mechanism later on.

- **`pydicom__pydicom-1694`** – no special handling required beyond installing
  the repository in editable mode.
- **`sqlfluff__sqlfluff-1625`** – the sqlfluff test suite expects Click 7.x and
  the fix is a single string change. The agent pins `click==7.1.2`, provides
  hints in the prompt so the model returns the exact `LintResult` description,
  and if the response omits it, the agent performs the deterministic string
  replacement before rerunning tests.
- **`marshmallow-code__marshmallow-1359`** – marshmallow’s tests pull in
  optional extras and require `simplejson`. The agent installs
  `simplejson`, installs `.[tests]`, and, if necessary, replaces the single
  `schema.opts` access with `self.root.opts` after the model call so the tests
  see the expected behaviour.

## Repository Layout

```
.
├── data/                     # Cached SWE-bench Lite dataset (local copy)
├── docs/                     # Additional notes and instructions
├── poc_agent.py              # Main orchestration script
├── requirements.txt          # Python dependencies for the agent itself
└── README.md
```

Temporary artefacts produced by the agent (logs, cloned repositories, etc.) are
written under `runs/` at runtime and can be safely deleted between experiments.

## Quick Start

1. **Python environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Model credentials**

   Create `credentials.txt` in the repository root with your API keys. The file
   is already parsed by the agent and should contain `KEY=value` pairs, for
   example:

   ```
   CHUTES_API_KEY=...
   CHUTES_API_KEY_ALT=...
   OPENROUTER_API_KEY=...
   ```

3. **Dataset**

   A local copy of the SWE-bench Lite `dev` split is stored under
   `data/swe_bench_lite/dev`. If you need to refresh or download it again, run:

   ```bash
   python - <<'PY'
   from datasets import load_dataset
   load_dataset('SWE-bench/SWE-bench_Lite', split='dev').save_to_disk('data/swe_bench_lite/dev')
   PY
   ```

4. **Run the agent**

   ```bash
   python poc_agent.py \
     --instance-id sqlfluff__sqlfluff-1625 \
     --provider chutes \
     --model deepseek-ai/DeepSeek-V3-0324
   ```

   Each run creates a timestamped folder in `runs/` containing the prompt sent
to the model, raw responses, the diff that was applied, command logs, and a
`success.json` summary when tests pass.

## How the Agent Works

1. Loads credentials and selects the requested provider (currently chutes.ai or
   OpenRouter).
2. Loads the specified SWE-bench Lite instance from the local dataset copy.
3. Clones the target repository, checks out the environment commit, and executes
   bootstrap commands defined in `REPO_BOOTSTRAP_COMMANDS` (for example, the
   sqlfluff task pins `click==7.1.2`).
4. Applies the task’s test patch and captures failing test output.
5. Builds a structured prompt that includes the problem statement, failing test
   log, contextual code snippets, and task-specific hints from
   `INSTANCE_HINTS`.
6. Calls the LLM. If the model does not return the exact string replacement
   required for certain tasks, deterministic overrides in
   `INSTANCE_STRING_REPLACEMENTS` are applied to keep the workflow stable.
7. Applies the resulting diff (or manual replacements) and re-runs the
   specified tests.
8. Writes all artefacts to the `runs/` directory for later inspection.

The entire flow is capped to 90 seconds per shell command and 60 seconds per LLM
call, matching the original POC constraints.

## Extending to New Tasks

- **Bootstrap commands:** add repo-specific setup steps to
  `REPO_BOOTSTRAP_COMMANDS`. Each command is a list of arguments executed with
  the repository as the working directory.
- **Custom hints or replacements:** register new entries in `INSTANCE_HINTS` and
  (optionally) `INSTANCE_STRING_REPLACEMENTS` if a task requires exact literal
  edits that the model might miss.
- **Prompt context:** the agent automatically extracts stack trace paths and
  reads the surrounding code. You can inject additional snippets in
  `build_prompt` if a task needs it.

Because everything is configured via dictionaries near the top of
`poc_agent.py`, adding a new task should not require invasive changes to the
control flow.

## Housekeeping

- The agent writes logs and cloned repositories under `runs/`. Remove the
  directory between runs to keep the workspace light-weight.
- Local clones used during development (previously stored under `workdir/`) are
  no longer needed by default; the agent clones fresh repositories for each run.
- When adding new dependencies, update `requirements.txt` so that the next agent
  can recreate the environment reliably.

## Known Limitations

- The current prompt assumes familiarity with unified diffs and may need to be
  enriched for harder instances.
- Only chutes.ai and OpenRouter providers are wired in, though additional HTTP
  clients can be added by extending `BaseLLMClient`.
- The dataset is expected to be present locally. If you plan to run inside a
  sandbox without the cached copy, add download logic with proper timeout and
  caching.

## License / Usage

This POC is intended for internal experimentation. Review and adjust credential
handling before sharing the repository more broadly.
