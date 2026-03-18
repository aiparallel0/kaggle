# =============================================================================
# diagnostics.py
# Purpose: Runtime AI-powered diagnostics for the DONUT SROIE pipeline.
#
# Four layers of autonomous error detection:
#   1. DiagnosticCallback  — TrainerCallback that detects known bug patterns
#                            during training (F1 collapse, loss plateau, OOM)
#   2. AIDiagnostics       — Calls Claude or Mistral API to interpret failures
#   3. RuntimeCheckpoint   — Structured telemetry collected per-epoch
#   4. PipelineDiagnostics — Stage-level wrapper for run_all.py orchestrator
#
# Usage:
#   from diagnostics import DiagnosticCallback, PipelineDiagnostics
#   # In train.py callback list:
#   callbacks.append(DiagnosticCallback(experiment_id=6))
#   # In run_all.py stage wrapper:
#   diag = PipelineDiagnostics(); diag.wrap_stage("DONUT Experiments", func)
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. RuntimeCheckpoint — structured telemetry per training event
# ---------------------------------------------------------------------------


@dataclass
class RuntimeCheckpoint:
    """Structured snapshot of pipeline state at a diagnostic checkpoint."""

    timestamp: float = 0.0
    stage: str = ""
    experiment_id: int = 0
    epoch: float = 0.0
    global_step: int = 0
    train_loss: float | None = None
    eval_loss: float | None = None
    eval_f1: float | None = None
    gpu_mem_allocated_mb: float | None = None
    gpu_mem_reserved_mb: float | None = None
    ram_used_mb: float | None = None
    issues: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != [] and v != {}}


# ---------------------------------------------------------------------------
# Known bug pattern signatures from CLAUDE.md §16
# ---------------------------------------------------------------------------

_BUG_PATTERNS: list[dict] = [
    {
        "name": "lm_head_dedup",
        "description": "safetensors deduplication dropped lm_head.weight (F1~0.42)",
        "detect": lambda cp: 0.40 < (cp.eval_f1 or 1.0) < 0.44 and (cp.epoch or 0) > 2,
        "severity": "critical",
        "fix": (
            "Verify LmHeadCloneCallback is registered. Check that "
            "config.tie_word_embeddings=False after resize_token_embeddings()."
        ),
    },
    {
        "name": "token2json_list",
        "description": "token2json returned list instead of dict (F1~0.008)",
        "detect": lambda cp: (cp.eval_f1 or 1.0) < 0.02 and (cp.epoch or 0) > 2,
        "severity": "critical",
        "fix": (
            "Model is emitting <sep/> tokens (CORD pretraining artifact). "
            "Ensure _parse_prediction() merges list pages into flat dict."
        ),
    },
    {
        "name": "total_f1_collapse",
        "description": "F1 dropped to exactly 0.0 — complete prediction failure",
        "detect": lambda cp: cp.eval_f1 == 0.0 and (cp.epoch or 0) > 1,
        "severity": "critical",
        "fix": (
            "Check decoder_start_token_id roundtrips to '<s_sroie>'. "
            "Check that convert_tokens_to_ids uses list form ['<s_sroie>']."
        ),
    },
    {
        "name": "loss_plateau",
        "description": "Training loss stuck above 2.0 after epoch 3 — underfitting",
        "detect": lambda cp: (cp.train_loss or 0) > 2.0 and (cp.epoch or 0) > 3,
        "severity": "warning",
        "fix": (
            "Too few optimizer steps (GP-2) or learning rate too low. "
            "Check validate_training_config() minimum 200 steps."
        ),
    },
    {
        "name": "loss_nan",
        "description": "Training loss is NaN — numerical instability",
        "detect": lambda cp: cp.train_loss is not None and cp.train_loss != cp.train_loss,
        "severity": "critical",
        "fix": (
            "Likely fp16 overflow. Switch to bf16 (Ampere+) or fp32. "
            "Check gradient clipping (max_grad_norm=1.0)."
        ),
    },
    {
        "name": "gpu_oom_warning",
        "description": "GPU memory >90% utilized — OOM risk",
        "detect": lambda cp: (
            (cp.gpu_mem_allocated_mb or 0) > 0
            and (cp.gpu_mem_reserved_mb or 1) > 0
            and (cp.gpu_mem_allocated_mb or 0) / max(cp.gpu_mem_reserved_mb or 1, 1) > 0.9
        ),
        "severity": "warning",
        "fix": "Reduce batch_size or enable gradient checkpointing.",
    },
    {
        "name": "eval_loss_diverging",
        "description": "Eval loss increasing while train loss decreasing — overfitting",
        "detect": lambda cp: False,  # needs history, handled in DiagnosticCallback
        "severity": "warning",
        "fix": "Consider early stopping or increasing weight_decay.",
    },
]


# ---------------------------------------------------------------------------
# 2. DiagnosticCallback — TrainerCallback for HuggingFace Seq2SeqTrainer
# ---------------------------------------------------------------------------


class DiagnosticCallback:
    """Runtime diagnostic callback that detects known bug patterns during training.

    Registered as a HuggingFace TrainerCallback. At each evaluation step,
    collects a RuntimeCheckpoint with GPU/RAM telemetry and checks it against
    known failure signatures from CLAUDE.md §16.

    Issues are logged immediately and accumulated in ``self.checkpoints`` for
    post-training analysis.

    Parameters
    ----------
    experiment_id : int
        The experiment number (1-8) for labeling.
    output_dir : Path or str or None
        Directory to write diagnostics JSON. Defaults to ``results/``.
    ai_diagnose : bool
        If True, call AI API for critical issues (requires API key).
    ai_provider : str
        AI provider: "claude", "mistral", or "auto" (tries claude then mistral).
    """

    def __init__(
        self,
        experiment_id: int = 0,
        output_dir: Path | str | None = None,
        ai_diagnose: bool = False,
        ai_provider: str = "auto",
        github_notify: bool = False,
        github_repo: str | None = None,
        github_issue_number: int | None = None,
    ):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir or "results")
        self.ai_diagnose = ai_diagnose
        self.ai_provider = ai_provider
        # GitHub notification: auto-enable when token + repo are present
        self.github_notify = github_notify or bool(_get_github_token() and _get_github_repo())
        self.github_repo = github_repo or _get_github_repo() or None
        self.github_issue_number = github_issue_number
        self.checkpoints: list[RuntimeCheckpoint] = []
        self._eval_loss_history: list[float] = []
        self._train_loss_history: list[float] = []
        self._issues_raised: set[str] = set()
        self._github_notified: set[str] = set()  # avoid duplicate issues per pattern

        # Dynamically inherit from TrainerCallback at instantiation
        # (avoids import-time dependency on transformers)
        try:
            from transformers import TrainerCallback

            if not isinstance(self, TrainerCallback):
                self.__class__ = type(
                    "DiagnosticCallback",
                    (self.__class__, TrainerCallback),
                    {},
                )
        except ImportError:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture training loss from log events."""
        if logs and "loss" in logs:
            self._train_loss_history.append(logs["loss"])
        return control

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        """Run diagnostic checks after each evaluation."""
        cp = self._collect_checkpoint(state, metrics)
        self.checkpoints.append(cp)

        # Run known-pattern detection
        issues = self._detect_patterns(cp)

        # Check eval-loss divergence (needs history)
        if metrics:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                self._eval_loss_history.append(eval_loss)
                if len(self._eval_loss_history) >= 3:
                    recent = self._eval_loss_history[-3:]
                    if recent[-1] > recent[-2] > recent[-3]:
                        train_losses = (
                            self._train_loss_history[-3:]
                            if len(self._train_loss_history) >= 3
                            else []
                        )
                        if train_losses and train_losses[-1] < train_losses[0]:
                            issues.append(
                                {
                                    "pattern": "eval_loss_diverging",
                                    "severity": "warning",
                                    "evidence": (
                                        f"Eval loss rising 3 consecutive evals: "
                                        f"{[f'{x:.4f}' for x in recent]} while train loss decreasing"
                                    ),
                                    "fix": "Consider early stopping or increasing weight_decay.",
                                }
                            )

        if issues:
            cp.issues = issues
            self._log_issues(cp, issues)

            # AI diagnosis for critical issues
            critical = [i for i in issues if i.get("severity") == "critical"]
            diagnosis: str | None = None
            if critical and self.ai_diagnose:
                self._run_ai_diagnosis(cp, critical)
                diagnosis = cp.metrics.get("ai_diagnosis")

            # GitHub notification — one issue per unique pattern per experiment
            if critical and self.github_notify:
                for issue in critical:
                    pattern_name = issue.get("pattern", "unknown")
                    notify_key = f"{self.experiment_id}:{pattern_name}"
                    if notify_key not in self._github_notified:
                        self._github_notified.add(notify_key)
                        github_report_failure(
                            stage=f"Experiment {self.experiment_id} training (epoch {cp.epoch:.0f})",
                            error_type=pattern_name,
                            error_message=issue.get("description", ""),
                            context={
                                "evidence": issue.get("evidence"),
                                "fix": issue.get("fix"),
                                "checkpoint": cp.to_dict(),
                            },
                            ai_diagnosis=diagnosis,
                            repo=self.github_repo,
                            issue_number=self.github_issue_number,
                        )

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Save diagnostic report at end of training."""
        self._save_report()
        return control

    def _collect_checkpoint(self, state, metrics) -> RuntimeCheckpoint:
        """Collect telemetry into a RuntimeCheckpoint."""
        cp = RuntimeCheckpoint(
            timestamp=time.time(),
            stage="training",
            experiment_id=self.experiment_id,
            epoch=getattr(state, "epoch", 0),
            global_step=getattr(state, "global_step", 0),
        )

        # Latest train loss
        if self._train_loss_history:
            cp.train_loss = self._train_loss_history[-1]

        # Eval metrics
        if metrics:
            cp.eval_loss = metrics.get("eval_loss")
            cp.eval_f1 = metrics.get("eval_f1")
            cp.metrics = {k: v for k, v in metrics.items() if isinstance(v, int | float)}

        # GPU telemetry
        try:
            import torch

            if torch.cuda.is_available():
                cp.gpu_mem_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                cp.gpu_mem_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        except ImportError:
            pass

        # RAM telemetry
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Only set ram_used_mb when total is also available
                        cp.ram_used_mb = None
                        break
        except (OSError, ValueError):
            pass

        return cp

    def _detect_patterns(self, cp: RuntimeCheckpoint) -> list[dict]:
        """Check checkpoint against known bug pattern signatures."""
        issues = []
        for pattern in _BUG_PATTERNS:
            try:
                if pattern["detect"](cp):
                    pattern_name = pattern["name"]
                    # Only raise each pattern once per training run
                    if pattern_name not in self._issues_raised:
                        self._issues_raised.add(pattern_name)
                        issues.append(
                            {
                                "pattern": pattern_name,
                                "severity": pattern["severity"],
                                "description": pattern["description"],
                                "evidence": self._format_evidence(cp, pattern_name),
                                "fix": pattern["fix"],
                            }
                        )
            except Exception:
                pass  # pattern lambda failed — skip
        return issues

    def _format_evidence(self, cp: RuntimeCheckpoint, pattern_name: str) -> str:
        """Build human-readable evidence string for an issue."""
        parts = [f"epoch={cp.epoch:.1f}", f"step={cp.global_step}"]
        if cp.train_loss is not None:
            parts.append(f"train_loss={cp.train_loss:.4f}")
        if cp.eval_loss is not None:
            parts.append(f"eval_loss={cp.eval_loss:.4f}")
        if cp.eval_f1 is not None:
            parts.append(f"eval_f1={cp.eval_f1:.4f}")
        if cp.gpu_mem_allocated_mb is not None:
            parts.append(f"gpu_mem={cp.gpu_mem_allocated_mb:.0f}MB")
        return ", ".join(parts)

    def _log_issues(self, cp: RuntimeCheckpoint, issues: list[dict]):
        """Log detected issues with appropriate severity."""
        for issue in issues:
            severity = issue.get("severity", "warning")
            msg = (
                f"[Diagnostics] Exp {self.experiment_id} | "
                f"PATTERN DETECTED: {issue['pattern']} ({severity})\n"
                f"  Description: {issue['description']}\n"
                f"  Evidence: {issue['evidence']}\n"
                f"  Suggested fix: {issue['fix']}"
            )
            if severity == "critical":
                logger.error(msg)
                print(f"\n{'!' * 72}\n{msg}\n{'!' * 72}")
            else:
                logger.warning(msg)
                print(f"\n[WARN] {msg}")

    def _run_ai_diagnosis(self, cp: RuntimeCheckpoint, issues: list[dict]):
        """Call AI API (Claude or Mistral) for deeper diagnosis of critical issues."""
        context = {
            "experiment_id": self.experiment_id,
            "checkpoint": cp.to_dict(),
            "issues": issues,
            "eval_loss_history": self._eval_loss_history[-10:],
            "train_loss_history": self._train_loss_history[-20:],
        }

        diagnosis = ai_diagnose(
            context,
            provider=self.ai_provider,
        )

        if diagnosis:
            logger.info("[AI Diagnosis] Exp %d:\n%s", self.experiment_id, diagnosis)
            print(f"\n[AI Diagnosis] Exp {self.experiment_id}:\n{diagnosis}")

            # Attach to checkpoint
            cp.metrics["ai_diagnosis"] = diagnosis

    def _save_report(self):
        """Write accumulated diagnostics to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / f"diagnostics_exp{self.experiment_id}.json"

        report = {
            "experiment_id": self.experiment_id,
            "total_checkpoints": len(self.checkpoints),
            "issues_detected": sum(len(cp.issues) for cp in self.checkpoints),
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "summary": self._build_summary(),
        }

        report_path.write_text(json.dumps(report, indent=2))
        logger.info("Diagnostics report saved → %s", report_path)

    def _build_summary(self) -> dict:
        """Build a concise summary of all detected issues."""
        all_issues = []
        for cp in self.checkpoints:
            all_issues.extend(cp.issues)

        patterns = {}
        for issue in all_issues:
            name = issue.get("pattern", "unknown")
            if name not in patterns:
                patterns[name] = {
                    "count": 0,
                    "severity": issue.get("severity"),
                    "description": issue.get("description"),
                    "fix": issue.get("fix"),
                }
            patterns[name]["count"] += 1

        return {
            "total_issues": len(all_issues),
            "critical_count": sum(1 for i in all_issues if i.get("severity") == "critical"),
            "warning_count": sum(1 for i in all_issues if i.get("severity") == "warning"),
            "patterns": patterns,
            "health": "healthy"
            if not all_issues
            else (
                "degraded"
                if not any(i.get("severity") == "critical" for i in all_issues)
                else "critical"
            ),
        }


# ---------------------------------------------------------------------------
# 3. AI Diagnosis — Claude and Mistral API integration
# ---------------------------------------------------------------------------

_DIAGNOSIS_SYSTEM_PROMPT = """\
You are a diagnostic agent for a DONUT (Document Understanding Transformer) \
fine-tuning pipeline for receipt key information extraction (SROIE Task-3).

Known critical bug patterns:
- lm_head_dedup (F1~0.42): safetensors deduplication drops lm_head.weight after \
resize_token_embeddings(). Fix: LmHeadCloneCallback + tie_word_embeddings=False.
- token2json_list (F1~0.008): token2json returns list when <sep/> tokens emitted. \
Fix: merge list pages into flat dict in _parse_prediction().
- F1=0.0: wrong decoder_start_token_id. Fix: use list form for convert_tokens_to_ids.
- Loss plateau (>2.0 at epoch 3+): too few optimizer steps (<200). Fix: validate_training_config().
- Eval loss diverging: overfitting. Fix: early stopping or higher weight_decay.
- GPU OOM: batch too large. Fix: halve batch_size, double gradient_accumulation_steps.

Target fields: company, date, address, total.
Base model: naver-clova-ix/donut-base.
Best known F1: 0.8982 (Exp 6: SROIE + Invoices, 2x oversample).

Analyze the runtime context and provide:
1. Most likely root cause (reference the specific bug pattern if matching)
2. Concrete fix (file + line-level specificity)
3. Expected impact if not fixed
Keep response under 300 words."""


def _call_claude(
    context: dict,
    model: str = "claude-sonnet-4-5-20251001",
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> str | None:
    """Call Claude API for diagnosis. Returns response text or None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # Try reading from file
        for path in ["anthropic_api_key.txt", ".anthropic_key"]:
            try:
                api_key = Path(path).read_text().strip()
                if api_key:
                    break
            except OSError:
                continue
    if not api_key:
        logger.debug("No Anthropic API key found — skipping Claude diagnosis")
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or _DIAGNOSIS_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Diagnose this pipeline issue:\n\n{json.dumps(context, indent=2, default=str)}",
                }
            ],
        )
        return response.content[0].text
    except ImportError:
        logger.debug("anthropic package not installed — pip install anthropic")
        return None
    except Exception as exc:
        logger.warning("Claude API call failed: %s", exc)
        return None


def _call_mistral(
    context: dict,
    model: str = "mistral-small-latest",
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> str | None:
    """Call Mistral API for diagnosis. Returns response text or None."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        for path in ["mistral_api_key.txt", ".mistral_key"]:
            try:
                api_key = Path(path).read_text().strip()
                if api_key:
                    break
            except OSError:
                continue
    if not api_key:
        logger.debug("No Mistral API key found — skipping Mistral diagnosis")
        return None

    try:
        from mistralai import Mistral

        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or _DIAGNOSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Diagnose this pipeline issue:\n\n{json.dumps(context, indent=2, default=str)}",
                },
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except ImportError:
        logger.debug("mistralai package not installed — pip install mistralai")
        return None
    except Exception as exc:
        logger.warning("Mistral API call failed: %s", exc)
        return None


def _call_mistral_httpx(
    context: dict,
    model: str = "mistral-small-latest",
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> str | None:
    """Call Mistral API via raw HTTP (no SDK required). Returns response text or None."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        for path in ["mistral_api_key.txt", ".mistral_key"]:
            try:
                api_key = Path(path).read_text().strip()
                if api_key:
                    break
            except OSError:
                continue
    if not api_key:
        return None

    import urllib.request

    payload = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt or _DIAGNOSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Diagnose this pipeline issue:\n\n{json.dumps(context, indent=2, default=str)}",
                },
            ],
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("Mistral HTTP call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# GitHub API integration — create issues / post comments on failures
# ---------------------------------------------------------------------------


def _get_github_token() -> str:
    """Read GitHub token from env var or file. Returns empty string if absent."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        for path in ["github_token.txt", ".github_token"]:
            try:
                token = Path(path).read_text().strip()
                if token:
                    break
            except OSError:
                continue
    return token


def _get_github_repo() -> str:
    """Read target repo (owner/repo) from GITHUB_REPO env var or git remote."""
    repo = os.environ.get("GITHUB_REPO", "")
    if repo:
        return repo
    # Auto-detect from git remote origin
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
            if m:
                return m.group(1)
    except Exception:
        pass
    return ""


def _github_request(
    method: str, endpoint: str, token: str, payload: dict | None = None
) -> dict | None:
    """Execute a GitHub REST API call. Returns parsed JSON or None on failure.

    Parameters
    ----------
    method : str
        HTTP method — "GET", "POST", "PATCH".
    endpoint : str
        Path after ``https://api.github.com``, e.g. ``/repos/owner/repo/issues``.
    token : str
        GitHub personal-access token (``repo`` scope) or fine-grained token
        with Issues write permission.
    payload : dict or None
        Request body; serialised to JSON when given.
    """
    import urllib.request

    url = f"https://api.github.com{endpoint}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "donut-sroie-diagnostics/1.0",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("GitHub API call %s %s failed: %s", method, endpoint, exc)
        return None


def github_create_issue(
    title: str,
    body: str,
    labels: list[str] | None = None,
    repo: str | None = None,
) -> dict | None:
    """Create a GitHub issue and return the response dict (includes ``html_url``).

    Parameters
    ----------
    title : str
        Issue title (will be prefixed with ``[DONUT Pipeline]``).
    body : str
        Issue body in Markdown.
    labels : list[str] or None
        Optional label names to attach (must already exist in the repo).
    repo : str or None
        ``owner/repo`` slug.  Falls back to ``GITHUB_REPO`` env var.

    Returns
    -------
    dict or None
        GitHub issue object on success, or None if token / repo absent.
    """
    token = _get_github_token()
    if not token:
        logger.debug("No GitHub token — skipping issue creation")
        return None
    repo = repo or _get_github_repo()
    if not repo:
        logger.debug("No GITHUB_REPO set — skipping issue creation")
        return None

    payload: dict = {"title": f"[DONUT Pipeline] {title}", "body": body}
    if labels:
        payload["labels"] = labels

    result = _github_request("POST", f"/repos/{repo}/issues", token, payload)
    if result and "html_url" in result:
        logger.info("GitHub issue created: %s", result["html_url"])
        print(f"  [GitHub] Issue created: {result['html_url']}")
    return result


def github_post_comment(
    issue_number: int,
    body: str,
    repo: str | None = None,
) -> dict | None:
    """Post a comment on an existing GitHub issue or pull request.

    Parameters
    ----------
    issue_number : int
        Issue or PR number to comment on.
    body : str
        Comment body in Markdown.
    repo : str or None
        ``owner/repo`` slug.  Falls back to ``GITHUB_REPO`` env var.

    Returns
    -------
    dict or None
        GitHub comment object on success, or None on failure.
    """
    token = _get_github_token()
    if not token:
        logger.debug("No GitHub token — skipping comment post")
        return None
    repo = repo or _get_github_repo()
    if not repo:
        logger.debug("No GITHUB_REPO set — skipping comment post")
        return None

    result = _github_request(
        "POST",
        f"/repos/{repo}/issues/{issue_number}/comments",
        token,
        {"body": body},
    )
    if result and "html_url" in result:
        logger.info("GitHub comment posted: %s", result["html_url"])
        print(f"  [GitHub] Comment posted: {result['html_url']}")
    return result


def github_report_failure(
    stage: str,
    error_type: str,
    error_message: str,
    context: dict | None = None,
    ai_diagnosis: str | None = None,
    repo: str | None = None,
    issue_number: int | None = None,
) -> dict | None:
    """Create a GitHub issue (or comment on an existing one) for a pipeline failure.

    Composes a structured Markdown body from the failure context, any AI
    diagnosis text, and a reproduction hint.

    Parameters
    ----------
    stage : str
        Pipeline stage name (e.g. ``"Experiment 6"``).
    error_type : str
        Exception class name.
    error_message : str
        Short exception message (first 500 chars used).
    context : dict or None
        Extra telemetry (gpu, epoch, F1, etc.) to include as a code block.
    ai_diagnosis : str or None
        AI-generated diagnosis text to include in the issue body.
    repo : str or None
        ``owner/repo`` slug; defaults to ``GITHUB_REPO`` env var.
    issue_number : int or None
        If set, post as a comment on this issue instead of creating a new one.

    Returns
    -------
    dict or None
        GitHub API response dict on success, or None on failure.
    """
    # Build Markdown body
    lines: list[str] = [
        f"## Pipeline Failure: {stage}",
        "",
        f"**Error:** `{error_type}: {error_message[:500]}`",
        "",
    ]

    if context:
        lines += [
            "### Runtime Context",
            "```json",
            json.dumps(context, indent=2, default=str)[:2000],
            "```",
            "",
        ]

    if ai_diagnosis:
        lines += [
            "### AI Diagnosis",
            ai_diagnosis,
            "",
        ]

    lines += [
        "### Reproduction",
        "```bash",
        "python run_all.py --experiment 1  # minimal repro",
        "python diagnostics.py --smoke-test",
        "```",
        "",
        "*Auto-generated by `diagnostics.py` — DONUT SROIE pipeline*",
    ]

    body = "\n".join(lines)
    title = f"{stage}: {error_type}"

    if issue_number is not None:
        return github_post_comment(issue_number, body, repo=repo)
    return github_create_issue(title, body, labels=["bug", "pipeline-failure"], repo=repo)


def ai_diagnose(
    context: dict,
    provider: str = "auto",
    claude_model: str = "claude-sonnet-4-5-20251001",
    mistral_model: str = "mistral-small-latest",
    system_prompt: str | None = None,
    max_tokens: int = 500,
) -> str | None:
    """Call an AI provider to diagnose a pipeline failure.

    Parameters
    ----------
    context : dict
        Runtime context (checkpoint data, issues, loss history, etc.).
    provider : str
        "claude", "mistral", or "auto" (tries claude, then mistral SDK,
        then mistral HTTP fallback).
    claude_model : str
        Claude model ID.
    mistral_model : str
        Mistral model ID.
    system_prompt : str or None
        Override the default diagnosis system prompt.
    max_tokens : int
        Maximum tokens in the AI response (default 500; use higher for code generation).

    Returns
    -------
    str or None
        Diagnosis text, or None if no API is available.
    """
    if provider == "claude":
        return _call_claude(
            context, model=claude_model, system_prompt=system_prompt, max_tokens=max_tokens
        )
    if provider == "mistral":
        result = _call_mistral(
            context, model=mistral_model, system_prompt=system_prompt, max_tokens=max_tokens
        )
        if result is None:
            result = _call_mistral_httpx(
                context, model=mistral_model, system_prompt=system_prompt, max_tokens=max_tokens
            )
        return result
    # auto: try claude first, then mistral
    result = _call_claude(
        context, model=claude_model, system_prompt=system_prompt, max_tokens=max_tokens
    )
    if result is not None:
        return result
    result = _call_mistral(
        context, model=mistral_model, system_prompt=system_prompt, max_tokens=max_tokens
    )
    if result is not None:
        return result
    return _call_mistral_httpx(
        context, model=mistral_model, system_prompt=system_prompt, max_tokens=max_tokens
    )


# ---------------------------------------------------------------------------
# 4. PipelineDiagnostics — stage-level diagnostics for run_all.py
# ---------------------------------------------------------------------------


class PipelineDiagnostics:
    """Collects diagnostics across all pipeline stages.

    Wraps each stage execution to capture exceptions, timing, GPU state,
    and optionally calls AI diagnosis on failures.

    Usage::

        diag = PipelineDiagnostics(ai_diagnose=True, ai_provider="auto")
        result = diag.wrap_stage("DONUT Experiments", stage_experiments, args)
        # At end of pipeline:
        diag.save_report()
    """

    def __init__(
        self,
        ai_diagnose: bool = False,
        ai_provider: str = "auto",
        output_dir: Path | str | None = None,
        github_notify: bool = False,
        github_repo: str | None = None,
        github_issue_number: int | None = None,
    ):
        self.ai_diagnose = ai_diagnose
        self.ai_provider = ai_provider
        self.output_dir = Path(output_dir or "results")
        self.stage_reports: list[dict] = []
        # GitHub notification: create issue or comment on failure
        self.github_notify = github_notify or bool(_get_github_token() and _get_github_repo())
        self.github_repo = github_repo or _get_github_repo() or None
        self.github_issue_number = github_issue_number

    def wrap_stage(self, stage_name: str, func, args) -> object:
        """Execute a stage function with diagnostic wrapping.

        Returns the stage's return value (typically a StageResult).
        On exception, captures full context and optionally runs AI diagnosis.
        """
        t0 = time.monotonic()
        gpu_before = self._gpu_snapshot()

        try:
            result = func(args)
            elapsed = time.monotonic() - t0
            gpu_after = self._gpu_snapshot()

            report = {
                "stage": stage_name,
                "status": "success",
                "duration_sec": elapsed,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
            }
            self.stage_reports.append(report)
            return result

        except Exception as exc:
            elapsed = time.monotonic() - t0
            gpu_after = self._gpu_snapshot()
            tb = traceback.format_exc()

            report = {
                "stage": stage_name,
                "status": "failed",
                "duration_sec": elapsed,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
            }
            self.stage_reports.append(report)

            logger.error(
                "[PipelineDiagnostics] Stage %r failed after %.1fs: %s: %s",
                stage_name,
                elapsed,
                type(exc).__name__,
                exc,
            )

            # AI diagnosis on failure
            diagnosis: str | None = None
            if self.ai_diagnose:
                diagnosis = ai_diagnose(
                    {
                        "stage": stage_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": tb[-2000:],  # truncate long tracebacks
                        "gpu_state": gpu_after,
                    },
                    provider=self.ai_provider,
                )
                if diagnosis:
                    report["ai_diagnosis"] = diagnosis
                    logger.info(
                        "[AI Diagnosis] Stage %r:\n%s",
                        stage_name,
                        diagnosis,
                    )
                    print(f"\n[AI Diagnosis] Stage {stage_name!r}:\n{diagnosis}")

            # GitHub notification on failure
            if self.github_notify:
                gh_result = github_report_failure(
                    stage=stage_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    context={
                        "traceback": tb[-1500:],
                        "duration_sec": round(elapsed, 1),
                        "gpu_state": gpu_after,
                    },
                    ai_diagnosis=diagnosis,
                    repo=self.github_repo,
                    issue_number=self.github_issue_number,
                )
                if gh_result:
                    report["github_url"] = gh_result.get("html_url")

            raise

    def save_report(self) -> Path:
        """Save the accumulated pipeline diagnostic report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "pipeline_diagnostics.json"

        report = {
            "total_stages": len(self.stage_reports),
            "failed_stages": sum(1 for r in self.stage_reports if r["status"] == "failed"),
            "stages": self.stage_reports,
        }
        report_path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Pipeline diagnostics saved → %s", report_path)
        return report_path

    @staticmethod
    def _gpu_snapshot() -> dict | None:
        """Capture current GPU memory state."""
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                }
        except ImportError:
            pass
        return None


# ---------------------------------------------------------------------------
# 5. Smoke test — fast validation of pipeline code paths
# ---------------------------------------------------------------------------


def smoke_test(verbose: bool = True) -> bool:
    """Run a fast (<30s) smoke test of the pipeline without real training.

    Validates:
    1. Import chain (constants, dataset_loaders)
    2. Model loads and forward pass succeeds (1 sample)
    3. Tokenizer special tokens are configured correctly
    4. decoder_start_token_id roundtrips correctly (GP-4)
    5. convert_tokens_to_ids uses list form (GP-3)

    Returns True if all checks pass, False otherwise.
    """
    checks_passed = 0
    checks_total = 0
    errors = []

    def _check(name: str, func):
        nonlocal checks_passed, checks_total
        checks_total += 1
        try:
            func()
            checks_passed += 1
            if verbose:
                print(f"  [PASS] {name}")
        except Exception as exc:
            errors.append((name, str(exc)))
            if verbose:
                print(f"  [FAIL] {name}: {exc}")

    if verbose:
        print("=" * 60)
        print("PIPELINE SMOKE TEST")
        print("=" * 60)

    # 1. Import chain
    def _check_imports():
        from constants import BASE_MODEL, FIELDS, NEW_TOKENS, SEED  # noqa: F401

    _check("Import constants", _check_imports)

    def _check_dataset_loaders():
        from data_pipeline import SROIELoader  # noqa: F401

    _check("Import dataset loaders", _check_dataset_loaders)

    # 2. Model + processor load
    def _check_model_load():
        from constants import BASE_MODEL, NEW_TOKENS

        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise RuntimeError(f"transformers not installed: {e}") from e

        processor = DonutProcessor.from_pretrained(BASE_MODEL)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        model.config.tie_word_embeddings = False

    _check("Model + processor load", _check_model_load)

    # 3. Token ID roundtrip (GP-3 + GP-4)
    def _check_token_ids():
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        from constants import BASE_MODEL, NEW_TOKENS

        processor = DonutProcessor.from_pretrained(BASE_MODEL)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

        # GP-3: list form
        token_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        assert token_id != processor.tokenizer.unk_token_id, (
            f"<s_sroie> mapped to unk_token_id={processor.tokenizer.unk_token_id}"
        )

        # GP-4: roundtrip
        decoded = processor.tokenizer.decode([token_id])
        assert "<s_sroie>" in decoded, f"Roundtrip failed: ID {token_id} decoded to {decoded!r}"

    _check("Token ID roundtrip (GP-3/GP-4)", _check_token_ids)

    # 4. Forward pass (dummy input)
    def _check_forward():
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        from constants import BASE_MODEL, NEW_TOKENS

        processor = DonutProcessor.from_pretrained(BASE_MODEL)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
        model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        model.config.tie_word_embeddings = False
        model.eval()

        # Create dummy image (3x1280x960 random tensor → pixel_values)
        from PIL import Image

        dummy_img = Image.new("RGB", (960, 1280), color=(128, 128, 128))
        inputs = processor(dummy_img, return_tensors="pt")

        # Create dummy decoder input
        start_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        decoder_input_ids = torch.tensor([[start_id]])

        with torch.no_grad():
            output = model(
                pixel_values=inputs.pixel_values,
                decoder_input_ids=decoder_input_ids,
            )
        assert output.logits.shape[-1] == len(processor.tokenizer), (
            f"Output vocab size {output.logits.shape[-1]} != tokenizer size {len(processor.tokenizer)}"
        )

    _check("Forward pass (dummy input)", _check_forward)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Results: {checks_passed}/{checks_total} passed")
        if errors:
            print("Failures:")
            for name, err in errors:
                print(f"  - {name}: {err}")
        print("=" * 60)

    return checks_passed == checks_total


# ---------------------------------------------------------------------------
# 6. Preflight diagnostic runner
# ---------------------------------------------------------------------------


def run_preflight(ai_diagnose_on_failure: bool = False, ai_provider: str = "auto") -> bool:
    """Quick preflight check that runs before experiments.

    Runs the smoke test and, if it fails and AI diagnosis is enabled,
    sends the failure context to an AI for root-cause analysis.

    Returns True if all checks pass.
    """
    print("\n[Preflight] Running diagnostic checks...")
    passed = smoke_test(verbose=True)

    if not passed and ai_diagnose_on_failure:
        print("\n[Preflight] Smoke test failed — requesting AI diagnosis...")
        diagnosis = ai_diagnose(
            {
                "stage": "preflight_smoke_test",
                "status": "failed",
                "note": "One or more smoke test checks failed. See output above.",
            },
            provider=ai_provider,
        )
        if diagnosis:
            print(f"\n[AI Diagnosis]:\n{diagnosis}")

    return passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DONUT Pipeline Diagnostics")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke test (<30s)",
    )
    parser.add_argument(
        "--ai-diagnose",
        action="store_true",
        help="Enable AI-powered diagnosis (requires API key)",
    )
    parser.add_argument(
        "--ai-provider",
        choices=["claude", "mistral", "auto"],
        default="auto",
        help="AI provider for diagnosis (default: auto)",
    )
    parser.add_argument(
        "--github-notify",
        action="store_true",
        help="Post smoke-test failures as GitHub issues (requires GITHUB_TOKEN + GITHUB_REPO)",
    )
    parser.add_argument(
        "--github-repo",
        default=None,
        metavar="OWNER/REPO",
        help="GitHub repo slug, e.g. aiparallel0/kaggle (overrides GITHUB_REPO env var)",
    )
    parser.add_argument(
        "--github-issue",
        type=int,
        default=None,
        metavar="N",
        help="Post as a comment on issue/PR N instead of creating a new issue",
    )
    parser.add_argument(
        "--github-test",
        action="store_true",
        help="Test GitHub API connectivity: create a test issue and immediately close it",
    )
    args = parser.parse_args()

    # Override env var when --github-repo is supplied
    if args.github_repo:
        os.environ["GITHUB_REPO"] = args.github_repo

    # GitHub connectivity test
    if args.github_test:
        print("[GitHub Test] Creating test issue...")
        result = github_create_issue(
            title="API connectivity test",
            body=(
                "This issue was created by `python diagnostics.py --github-test` "
                "to verify GitHub API connectivity.\n\n"
                "It can be closed immediately."
            ),
            labels=[],
        )
        if result:
            print(f"[GitHub Test] PASS — issue created: {result.get('html_url')}")
            # Immediately close it
            token = _get_github_token()
            repo = _get_github_repo()
            if token and repo:
                issue_num = result.get("number")
                _github_request(
                    "PATCH",
                    f"/repos/{repo}/issues/{issue_num}",
                    token,
                    {"state": "closed"},
                )
                print(f"[GitHub Test] Issue #{issue_num} closed.")
        else:
            print("[GitHub Test] FAIL — check GITHUB_TOKEN and GITHUB_REPO")
        raise SystemExit(0 if result else 1)

    if args.smoke_test:
        ok = run_preflight(
            ai_diagnose_on_failure=args.ai_diagnose,
            ai_provider=args.ai_provider,
        )
        if not ok and (args.github_notify or _get_github_token()):
            github_report_failure(
                stage="smoke_test",
                error_type="SmokeTestFailure",
                error_message="One or more preflight checks failed — see console output.",
                repo=args.github_repo,
                issue_number=args.github_issue,
            )
        raise SystemExit(0 if ok else 1)

    # Default: just run smoke test
    ok = run_preflight(
        ai_diagnose_on_failure=args.ai_diagnose,
        ai_provider=args.ai_provider,
    )
    raise SystemExit(0 if ok else 1)
