"""Central configuration for Denial Defense."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    wandb_api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))
    weave_project: str = field(
        default_factory=lambda: os.getenv("WEAVE_PROJECT", "denial-defense")
    )
    agent_model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "gpt-4o"))
    supervisor_model: str = field(
        default_factory=lambda: os.getenv("SUPERVISOR_MODEL", os.getenv("AGENT_MODEL", "gpt-4o"))
    )
    scorer_model: str = field(
        default_factory=lambda: os.getenv("SCORER_MODEL", "openai/gpt-oss-120b")
    )
    agent_temperature: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    )
    supervisor_temperature: float = field(
        default_factory=lambda: float(os.getenv("SUPERVISOR_TEMPERATURE", "0.4"))
    )
    max_rounds: int = field(default_factory=lambda: int(os.getenv("MAX_ROUNDS", "2")))
    session_token_budget: int = field(
        default_factory=lambda: int(os.getenv("SESSION_TOKEN_BUDGET", "120000"))
    )
    flask_host: str = field(default_factory=lambda: os.getenv("FLASK_HOST", "127.0.0.1"))
    flask_port: int = field(default_factory=lambda: int(os.getenv("FLASK_PORT", "5000")))
    flask_api_key: str = field(default_factory=lambda: os.getenv("FLASK_API_KEY", ""))
    flask_debug: bool = field(
        default_factory=lambda: os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )
    imr_data_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "IMR_DATA_PATH",
                str(ROOT / "data" / "raw" / "imr" / "independent-medical-review-imr-determinations-trend.csv"),
            )
        )
    )
    eval_parquet_path: Path = field(
        default_factory=lambda: ROOT / "data" / "processed" / "eval_set_imr.parquet"
    )
    playbook_path: Path = field(
        default_factory=lambda: ROOT / "data" / "processed" / "denial_playbook.json"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
