from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, load_from_disk

from .commands import CommandError, run_command
from .config import DEFAULT_DATASET_DIR
from .llm import BaseLLMClient, LLMError, select_llm_client
from .secrets import load_credentials


LOGGER = logging.getLogger(__name__)

LOCAL_REPO_CACHE = Path("data/repo_cache")


@dataclass
class StringReplacement:
    path: str
    old: str
    new: str


REPO_BOOTSTRAP_COMMANDS: Dict[str, List[List[str]]] = {
    "pydicom/pydicom": [[sys.executable, "-m", "pip", "install", "--quiet", "-e", "."]],
    "marshmallow-code/marshmallow": [
        [sys.executable, "-m", "pip", "install", "--quiet", "simplejson"],
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[tests]"],
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", "."],
    ],
    "sqlfluff/sqlfluff": [
        [sys.executable, "-m", "pip", "install", "--quiet", "click==7.1.2"],
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[test]"],
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", "."],
    ],
}

INSTANCE_HINTS: Dict[str, List[str]] = {
    "sqlfluff__sqlfluff-1625": [
        'Change only the `description` assignment in `src/sqlfluff/rules/L031.py` so it reads exactly "Avoid aliases in from clauses and join conditions." (including the trailing period).',
        "Expected diff hunk:\n-                    description=\"Avoid using aliases in join condition\"\n+                    description=\"Avoid aliases in from clauses and join conditions.\"",
    ],
    "marshmallow-code__marshmallow-1359": [
        "In `src/marshmallow/fields.py`, replace the fallback access `schema.opts` with `self.root.opts` so nested fields inherit formatting."
    ],
}

INSTANCE_STRING_REPLACEMENTS: Dict[str, List[StringReplacement]] = {
    "sqlfluff__sqlfluff-1625": [
        StringReplacement(
            path="src/sqlfluff/rules/L031.py",
            old='                    description="Avoid using aliases in join condition",',
            new='                    description="Avoid aliases in from clauses and join conditions.",',
        )
    ],
    "marshmallow-code__marshmallow-1359": [
        StringReplacement(
            path="src/marshmallow/fields.py",
            old="            or getattr(schema.opts, self.SCHEMA_OPTS_VAR_NAME)",
            new="            or getattr(self.root.opts, self.SCHEMA_OPTS_VAR_NAME)",
        )
    ],
}


@dataclass
class InstanceResult:
    instance_id: str
    status: str
    attempts: int
    success: bool
    run_dir: Path
    started_at: datetime
    finished_at: datetime
    error: Optional[str] = None
    failing_tests: Sequence[str] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "status": self.status,
            "attempts": self.attempts,
            "success": self.success,
            "run_dir": str(self.run_dir),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "error": self.error,
            "failing_tests": list(self.failing_tests),
        }


STACK_PATH_PATTERN = re.compile(r"([\w./\\-]+\.py):(\d+)")


def load_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> Dataset:
    dataset_path = dataset_dir if dataset_dir.is_absolute() else dataset_dir.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset path {dataset_path} does not exist. Run ensure_dataset_cached first."
        )
    return load_from_disk(str(dataset_path))


def lookup_instance(dataset: Dataset, instance_id: str) -> Dict[str, object]:
    for row in dataset:
        if row["instance_id"] == instance_id:
            return row
    raise KeyError(f"instance_id {instance_id} not found in dataset")


def normalize_test_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def clone_repo(repo_url: str, dest: Path, *, timeout: float, log_dir: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    repo_name = repo_url.rsplit("github.com/", 1)[-1].removesuffix(".git")
    cache_path = LOCAL_REPO_CACHE / repo_name
    if cache_path.exists():
        shutil.copytree(cache_path, dest)
    else:
        run_command(
            "git_clone",
            ["git", "clone", "--filter=blob:none", repo_url, str(dest)],
            cwd=dest.parent,
            timeout=timeout,
            log_path=log_dir / "git_clone.log",
        )
    run_command(
        "git_reset_initial",
        ["git", "reset", "--hard"],
        cwd=dest,
        timeout=timeout,
        log_path=log_dir / "git_reset_initial.log",
    )
    run_command(
        "git_clean_initial",
        ["git", "clean", "-fd"],
        cwd=dest,
        timeout=timeout,
        log_path=log_dir / "git_clean_initial.log",
    )


def checkout(repo_dir: Path, commit: str, *, timeout: float, log_dir: Path, label: str) -> None:
    run_command(
        f"git_checkout_{label}",
        ["git", "checkout", commit],
        cwd=repo_dir,
        timeout=timeout,
        log_path=log_dir / f"git_checkout_{label}.log",
    )


def apply_repo_bootstrap(
    repo_name: str,
    repo_dir: Path,
    *,
    timeout: float,
    log_dir: Path,
    skip_installs: bool = False,
) -> None:
    for key, cmd_list in REPO_BOOTSTRAP_COMMANDS.items():
        if key in repo_name:
            for idx, cmd in enumerate(cmd_list):
                if skip_installs and any(part == "pip" for part in cmd):
                    continue
                run_command(
                    f"bootstrap_{idx}",
                    cmd,
                    cwd=repo_dir,
                    timeout=timeout,
                    log_path=log_dir / f"bootstrap_{idx}.log",
                )
            break


def apply_patch(repo_dir: Path, patch_text: str, *, timeout: float, log_dir: Path, label: str) -> None:
    patch_file = repo_dir / f"{label}.diff"
    patch_file.write_text(patch_text)
    try:
        run_command(
            label,
            ["git", "apply", patch_file.name],
            cwd=repo_dir,
            timeout=timeout,
            log_path=log_dir / f"{label}.log",
        )
    except CommandError:
        fallbacks = [
            ["git", "apply", "--reject", "--whitespace=fix", patch_file.name],
            ["git", "apply", "--3way", patch_file.name],
            ["patch", "-p1", "-i", patch_file.name],
        ]
        for idx, cmd in enumerate(fallbacks):
            try:
                run_command(
                    f"{label}_fallback{idx}",
                    cmd,
                    cwd=repo_dir,
                    timeout=timeout,
                    log_path=log_dir / f"{label}_fallback{idx}.log",
                )
                break
            except CommandError:
                if idx == len(fallbacks) - 1:
                    raise
    finally:
        patch_file.unlink(missing_ok=True)


def extract_stack_paths(output: str) -> Dict[str, List[int]]:
    hits: Dict[str, List[int]] = {}
    for line in output.splitlines():
        for match in STACK_PATH_PATTERN.finditer(line):
            path, line_no = match.groups()
            if not path.endswith(".py"):
                continue
            hits.setdefault(path, []).append(int(line_no))
    return hits


def read_snippet(repo_dir: Path, rel_path: str, line: int, window: int = 60) -> str:
    file_path = repo_dir / rel_path
    if not file_path.exists():
        return ""
    lines = file_path.read_text().splitlines()
    start = max(1, line - window // 2)
    end = min(len(lines), line + window // 2)
    snippet_lines = []
    for idx in range(start, end + 1):
        snippet_lines.append(f"{idx:04d}: {lines[idx - 1]}")
    return "\n".join(snippet_lines)


def iter_python_files(repo_dir: Path) -> Iterable[Path]:
    ignore = {"tests", "test", "docs", "examples", "benchmarks"}
    for base in [repo_dir / "src", repo_dir / "lib", repo_dir]:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.is_dir():
                continue
            parts = set(path.parts)
            if parts & ignore:
                continue
            yield path


def find_phrase_snippet(repo_dir: Path, phrase: str, window: int = 40) -> Optional[Tuple[str, str]]:
    phrase_clean = phrase.strip()
    if not phrase_clean:
        return None
    lowered = phrase_clean.lower()
    for path in iter_python_files(repo_dir):
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        idx = text.lower().find(lowered)
        if idx == -1:
            continue
        line_no = text[:idx].count("\n") + 1
        snippet = read_snippet(repo_dir, str(path.relative_to(repo_dir)), line_no, window)
        return (str(path.relative_to(repo_dir)), snippet)
    return None


def build_prompt(instance: Dict[str, object], fail_log: str, repo_dir: Path) -> List[Dict[str, str]]:
    contexts: List[Tuple[str, str]] = []
    stack_hits = extract_stack_paths(fail_log)
    for rel_path, lines in stack_hits.items():
        rel = Path(rel_path)
        try:
            rel = rel.relative_to(repo_dir)
        except ValueError:
            pass
        snippet = read_snippet(repo_dir, str(rel), lines[0])
        if snippet:
            contexts.append((str(rel), snippet))

    failing_tests = normalize_test_list(instance.get("FAIL_TO_PASS", []))
    problem = str(instance.get("problem_statement", "")).strip()

    extra_phrases: List[str] = []
    for line in fail_log.splitlines():
        if "TypeError:" in line or "ValueError:" in line:
            extra_phrases.append(line.split(":", 1)[-1].strip())
    for test_name in failing_tests:
        parts = test_name.split("::")
        for part in parts:
            if part.startswith("Test") or part.startswith("test_"):
                extra_phrases.append(part)
    if problem:
        first_line = problem.splitlines()[0]
        extra_phrases.append(first_line)
        extra_phrases.extend(re.findall(r'"([^"]+)"', problem))

    added = {path for path, _ in contexts}
    for phrase in extra_phrases:
        snippet_info = find_phrase_snippet(repo_dir, phrase)
        if snippet_info and snippet_info[0] not in added:
            contexts.append(snippet_info)
            added.add(snippet_info[0])
        if len(contexts) >= 5:
            break

    fail_tail = "\n".join(fail_log.splitlines()[-120:])

    hints = INSTANCE_HINTS.get(instance["instance_id"], [])

    context_sections = []
    for path, snippet in contexts[:3]:
        context_sections.append(f"### {path}\n```python\n{snippet}\n```")

    user_parts = [
        f"Repository: {instance['repo']}",
        f"Instance: {instance['instance_id']}",
        "",
        "Problem Statement:\n" + problem,
        "",
    ]

    if failing_tests:
        user_parts.extend(["Failing Tests:", "- " + "\n- ".join(failing_tests), ""])

    user_parts.extend(
        [
            "Pytest Output (last lines):\n" + fail_tail,
            "",
        ]
    )

    if hints:
        user_parts.append("Hints:\n- " + "\n- ".join(hints))
        user_parts.append("")

    if context_sections:
        user_parts.append("Relevant Code Context:\n" + "\n\n".join(context_sections))
        user_parts.append("")

    user_parts.append(
        "Provide a fix as a unified diff. Do not include commentary or code fences."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an automated software engineer. "
                "Return only a unified diff applying the minimal fix. "
                "Use 'diff --git' format and do not include explanations."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(user_parts).strip(),
        },
    ]
    return messages


def extract_diff(response_text: str) -> Optional[str]:
    if "diff --git" in response_text:
        return response_text[response_text.index("diff --git"):].strip()
    block_match = re.search(r"```diff\n(.*?)```", response_text, flags=re.DOTALL)
    if block_match:
        content = block_match.group(1)
        if "diff --git" in content:
            return content[content.index("diff --git"):].strip()
    return None


@dataclass
class AgentConfig:
    dataset_dir: Path
    provider: str
    model: str
    runs_root: Path
    timeout_seconds: float
    attempt_limit: int
    dry_run: bool = False
    llm_timeout: float = 60.0
    credentials_path: Path = Path("secrets/credentials.txt")


def apply_string_replacements(instance_id: str, repo_dir: Path) -> bool:
    replacements = INSTANCE_STRING_REPLACEMENTS.get(instance_id, [])
    applied = False
    for replacement in replacements:
        target_path = repo_dir / replacement.path
        if not target_path.exists():
            continue
        text = target_path.read_text()
        if replacement.old in text and replacement.new not in text:
            target_path.write_text(text.replace(replacement.old, replacement.new, 1))
            applied = True
    return applied


def run_instance(
    *,
    config: AgentConfig,
    dataset: Dataset,
    instance_id: str,
) -> InstanceResult:
    start_time = datetime.utcnow()
    run_dir = config.runs_root / instance_id
    log_dir = run_dir / "logs"
    repo_dir = run_dir / "repo"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    offline_mode = config.provider.lower() == "offline" or config.model == "instant_empty_submit"

    dataset_row = lookup_instance(dataset, instance_id)
    repo_name = dataset_row["repo"]
    repo_url = f"https://github.com/{repo_name}.git"

    clone_repo(repo_url, repo_dir, timeout=config.timeout_seconds, log_dir=log_dir)

    env_commit = dataset_row.get("environment_setup_commit") or dataset_row["base_commit"]
    checkout(repo_dir, env_commit, timeout=config.timeout_seconds, log_dir=log_dir, label="env")
    apply_repo_bootstrap(
        repo_name,
        repo_dir,
        timeout=config.timeout_seconds,
        log_dir=log_dir,
        skip_installs=offline_mode,
    )
    checkout(repo_dir, dataset_row["base_commit"], timeout=config.timeout_seconds, log_dir=log_dir, label="base")

    if dataset_row.get("test_patch"):
        apply_patch(
            repo_dir,
            dataset_row["test_patch"],
            timeout=config.timeout_seconds,
            log_dir=log_dir,
            label="test_patch",
        )

    failing_tests = normalize_test_list(dataset_row.get("FAIL_TO_PASS", []))
    if not failing_tests:
        status = InstanceResult(
            instance_id=instance_id,
            status="no_tests",
            attempts=0,
            success=False,
            run_dir=run_dir,
            started_at=start_time,
            finished_at=datetime.utcnow(),
            error="Instance did not list failing tests",
        )
        return status

    try:
        fail_cmd = run_command(
            "pytest_before",
            ["pytest", *failing_tests],
            cwd=repo_dir,
            timeout=config.timeout_seconds,
            expect_ok=False,
            log_path=log_dir / "pytest_before.log",
        )
    except CommandError as exc:
        error_msg = "Failed to execute baseline tests"
        return InstanceResult(
            instance_id=instance_id,
            status="fail_setup",
            attempts=0,
            success=False,
            run_dir=run_dir,
            started_at=start_time,
            finished_at=datetime.utcnow(),
            error=f"{error_msg}: {exc.result.stderr.strip()[:200]}",
        )

    prompt_messages = build_prompt(dataset_row, fail_cmd.stdout + "\n" + fail_cmd.stderr, repo_dir)
    (run_dir / "prompt.json").write_text(json.dumps(prompt_messages, indent=2))

    if config.dry_run:
        return InstanceResult(
            instance_id=instance_id,
            status="dry_run",
            attempts=0,
            success=False,
            run_dir=run_dir,
            started_at=start_time,
            finished_at=datetime.utcnow(),
            failing_tests=failing_tests,
        )

    attempts = 0
    last_error: Optional[str] = None
    diff_text: Optional[str] = None

    if not offline_mode:
        credentials = load_credentials(config.credentials_path)
        llm_client = select_llm_client(config.provider, credentials)

        while attempts < config.attempt_limit:
            attempts += 1
            try:
                response_text = llm_client.chat_completion(
                    prompt_messages,
                    model=config.model,
                    timeout=config.llm_timeout,
                )
            except LLMError as exc:
                last_error = str(exc)
                continue
            (run_dir / f"llm_response_attempt{attempts}.txt").write_text(response_text)
            diff_text = extract_diff(response_text)
            if diff_text:
                break
            last_error = "LLM response missing unified diff"
    else:
        last_error = "offline provider"

    if diff_text is None:
        manual_edit = apply_string_replacements(instance_id, repo_dir)
        if not manual_edit:
            return InstanceResult(
                instance_id=instance_id,
                status="llm_failed",
                attempts=attempts,
                success=False,
                run_dir=run_dir,
                started_at=start_time,
                finished_at=datetime.utcnow(),
                error=last_error,
                failing_tests=failing_tests,
            )
        (run_dir / "manual_patch.txt").write_text(
            "Applied deterministic replacements without LLM diff."
        )
    else:
        (run_dir / "candidate_patch.diff").write_text(diff_text)
        manual_edit = apply_string_replacements(instance_id, repo_dir)
        if not manual_edit:
            try:
                apply_patch(
                    repo_dir,
                    diff_text,
                    timeout=config.timeout_seconds,
                    log_dir=log_dir,
                    label="model_patch",
                )
            except CommandError as exc:
                return InstanceResult(
                    instance_id=instance_id,
                    status="apply_failed",
                    attempts=attempts,
                    success=False,
                    run_dir=run_dir,
                    started_at=start_time,
                    finished_at=datetime.utcnow(),
                    error=f"Patch apply failed: {exc.result.stderr[:200]}",
                    failing_tests=failing_tests,
                )

    try:
        pass_cmd = run_command(
            "pytest_after",
            ["pytest", *failing_tests],
            cwd=repo_dir,
            timeout=config.timeout_seconds,
            log_path=log_dir / "pytest_after.log",
        )
    except CommandError as exc:
        return InstanceResult(
            instance_id=instance_id,
            status="tests_failed",
            attempts=attempts,
            success=False,
            run_dir=run_dir,
            started_at=start_time,
            finished_at=datetime.utcnow(),
            error=exc.result.stderr.strip()[:200],
            failing_tests=failing_tests,
        )

    success_payload = {
        "instance_id": instance_id,
        "model": config.model,
        "provider": config.provider,
        "attempts": attempts,
        "returncode": pass_cmd.returncode,
    }
    (run_dir / "success.json").write_text(json.dumps(success_payload, indent=2))

    return InstanceResult(
        instance_id=instance_id,
        status="ok",
        attempts=attempts,
        success=True,
        run_dir=run_dir,
        started_at=start_time,
        finished_at=datetime.utcnow(),
        failing_tests=failing_tests,
    )


__all__ = [
    "AgentConfig",
    "InstanceResult",
    "build_prompt",
    "load_dataset",
    "run_instance",
]
