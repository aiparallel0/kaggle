"""Cloud pipeline configuration management.

See also: training_config.py — TrainingConfig for model hyperparameters
          (learning rate, batch size, epochs, etc.).
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

__all__ = ["CloudConfig", "PipelineMode", "LogLevel"]


class PipelineMode(str, Enum):
    """Available pipeline execution modes."""

    CODE_REPAIR = "code_repair"
    ML_TRAINING = "ml_training"
    AUTO = "auto"


class LogLevel(str, Enum):
    """Log level options."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class CloudConfig:
    """Top-level configuration for cloud pipeline operations.

    All values can be overridden via environment variables:
    - CLOUD_PIPELINE_MODE
    - DONUT_WORKSPACE
    - GITHUB_REPO
    - GITHUB_BRANCH
    - OLLAMA_BASE_URL
    - OLLAMA_MODEL
    - etc.
    """

    # ====== Mode Selection ======
    mode: PipelineMode = PipelineMode.AUTO
    dry_run: bool = False

    # ====== Common Settings ======
    workspace: Path = Path("/workspace")  # overridden by DONUT_WORKSPACE env var
    git_branch: str = "claude/setup-cloud-ai-agents-Olrqd"
    github_repo: str = "aiparallel0/kaggle"
    skip_validation: bool = False

    # ====== Code Repair Settings (Ollama) ======
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    ollama_max_retries: int = 3
    ollama_auto_start: bool = True

    # ====== ML Training Settings (Vast.ai) ======
    gpu_required: bool = True
    skip_trocr: bool = False
    skip_pretrained_baseline: bool = False
    experiments_to_run: list[int] = field(default_factory=lambda: list(range(1, 9)))

    # ====== Cloud Storage Settings ======
    # Note: Based on user feedback, we commit results to GitHub (via git)
    # No S3/GCS needed, simplifies implementation
    s3_bucket: str | None = None
    s3_region: str | None = None
    gcs_bucket: str | None = None

    # ====== Validation Settings ======
    enable_ruff_check: bool = True
    enable_pytest: bool = True
    pytest_markers: str = ""
    fail_on_warnings: bool = False

    # ====== Commit Settings ======
    auto_commit: bool = True
    commit_on_error: bool = False

    # ====== Logging Settings ======
    log_level: LogLevel = LogLevel.INFO
    log_dir: Path = Path("./logs")
    stream_training_logs: bool = False  # Show training details in console

    # ====== Paths ======
    sroie_data_dir: Path | None = None
    results_dir: Path = Path("./results")

    @staticmethod
    def from_env() -> "CloudConfig":
        """Load configuration from environment variables with defaults."""

        # Determine mode
        mode_str = os.getenv("CLOUD_PIPELINE_MODE", "auto").lower()
        try:
            mode = PipelineMode(mode_str)
        except ValueError:
            mode = PipelineMode.AUTO

        # Workspace
        workspace = Path(os.getenv("DONUT_WORKSPACE", "/workspace"))

        # SROIE data directory
        sroie_dir = os.getenv("SROIE_DATA_DIR")
        if sroie_dir:
            sroie_dir = Path(sroie_dir)
        else:
            # Default to workspace/ICDAR-2019-SROIE/data
            sroie_dir = workspace / "ICDAR-2019-SROIE" / "data"

        # Ollama settings
        ollama_auto_start = os.getenv("OLLAMA_AUTO_START", "true").lower() == "true"

        # Results directory
        results_dir = Path(os.getenv("RESULTS_DIR", "./results"))

        # Log directory
        log_dir = Path(os.getenv("LOG_DIR", "./logs"))
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        try:
            log_level = LogLevel(log_level_str)
        except ValueError:
            log_level = LogLevel.INFO

        # Experiments to run
        exp_str = os.getenv("EXPERIMENTS_TO_RUN")
        if exp_str:
            try:
                experiments = [int(x.strip()) for x in exp_str.split(",")]
            except ValueError:
                experiments = list(range(1, 9))
        else:
            experiments = list(range(1, 9))

        # Validation settings
        enable_ruff = os.getenv("ENABLE_RUFF_CHECK", "true").lower() == "true"
        enable_pytest = os.getenv("ENABLE_PYTEST", "true").lower() == "true"
        fail_on_warnings = os.getenv("FAIL_ON_WARNINGS", "false").lower() == "true"

        # Git settings
        auto_commit = os.getenv("AUTO_COMMIT", "true").lower() == "true"
        commit_on_error = os.getenv("COMMIT_ON_ERROR", "false").lower() == "true"

        # Flags
        skip_trocr = os.getenv("SKIP_TROCR", "false").lower() == "true"
        skip_pretrained = os.getenv("SKIP_PRETRAINED_BASELINE", "false").lower() == "true"
        skip_validation = os.getenv("SKIP_VALIDATION", "false").lower() == "true"

        return CloudConfig(
            mode=mode,
            workspace=workspace,
            git_branch=os.getenv("GITHUB_BRANCH", "claude/setup-cloud-ai-agents-Olrqd"),
            github_repo=os.getenv("GITHUB_REPO", "aiparallel0/kaggle"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral:latest"),
            ollama_auto_start=ollama_auto_start,
            skip_trocr=skip_trocr,
            skip_pretrained_baseline=skip_pretrained,
            experiments_to_run=experiments,
            s3_bucket=os.getenv("AWS_S3_BUCKET"),
            s3_region=os.getenv("AWS_S3_REGION"),
            gcs_bucket=os.getenv("GCS_BUCKET"),
            enable_ruff_check=enable_ruff,
            enable_pytest=enable_pytest,
            fail_on_warnings=fail_on_warnings,
            auto_commit=auto_commit,
            commit_on_error=commit_on_error,
            sroie_data_dir=sroie_dir,
            results_dir=results_dir,
            log_dir=log_dir,
            log_level=log_level,
            skip_validation=skip_validation,
        )

    @staticmethod
    def from_args_and_env(args) -> "CloudConfig":
        """Load configuration from argparse args and environment."""
        config = CloudConfig.from_env()

        # Override with command-line arguments if provided
        if hasattr(args, "mode") and args.mode and args.mode != "auto":
            config.mode = PipelineMode(args.mode)

        if hasattr(args, "ollama_url"):
            config.ollama_base_url = args.ollama_url

        if hasattr(args, "ollama_model"):
            config.ollama_model = args.ollama_model

        if hasattr(args, "experiments") and args.experiments:
            config.experiments_to_run = args.experiments

        if hasattr(args, "skip_trocr") and args.skip_trocr:
            config.skip_trocr = True

        if hasattr(args, "s3_bucket"):
            config.s3_bucket = args.s3_bucket

        if hasattr(args, "gcs_bucket"):
            config.gcs_bucket = args.gcs_bucket

        if hasattr(args, "skip_validation") and args.skip_validation:
            config.skip_validation = True

        if hasattr(args, "dry_run") and args.dry_run:
            config.dry_run = True

        if hasattr(args, "no_commit") and args.no_commit:
            config.auto_commit = False

        if hasattr(args, "branch"):
            config.git_branch = args.branch

        if hasattr(args, "workspace"):
            config.workspace = Path(args.workspace)

        return config

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration for consistency.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Workspace must exist
        if not self.workspace.exists():
            errors.append(f"Workspace does not exist: {self.workspace}")

        # Results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # If ML training mode, SROIE data must exist
        if (
            self.mode in [PipelineMode.ML_TRAINING, PipelineMode.AUTO]
            and self.sroie_data_dir
            and not self.sroie_data_dir.exists()
        ):
            errors.append(f"SROIE data directory does not exist: {self.sroie_data_dir}")

        # Experiments must be in range 1-8
        for exp_id in self.experiments_to_run:
            if not (1 <= exp_id <= 8):
                errors.append(f"Invalid experiment ID: {exp_id} (must be 1-8)")

        return len(errors) == 0, errors
