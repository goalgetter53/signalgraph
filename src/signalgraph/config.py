"""Runtime settings, read from the environment (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONTACT = "Hussainsingapuri24@gmail.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path("signalgraph.duckdb"))
    contact_email: str = DEFAULT_CONTACT
    config_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "config")
    github_token: str | None = None
    http_timeout_s: float = 20.0
    max_retries: int = 4

    @property
    def user_agent(self) -> str:
        # SEC EDGAR fair-access policy requires a descriptive UA with a contact address.
        return f"signalgraph/0.1 (account-intelligence research; contact: {self.contact_email})"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=Path(os.environ.get("SIGNALGRAPH_DB", "signalgraph.duckdb")),
            contact_email=os.environ.get("SIGNALGRAPH_CONTACT_EMAIL", DEFAULT_CONTACT),
            config_dir=Path(os.environ.get("SIGNALGRAPH_CONFIG_DIR", str(PROJECT_ROOT / "config"))),
            github_token=os.environ.get("GITHUB_TOKEN") or None,
            http_timeout_s=float(os.environ.get("SIGNALGRAPH_HTTP_TIMEOUT", "20")),
            max_retries=int(os.environ.get("SIGNALGRAPH_MAX_RETRIES", "4")),
        )
