from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "INFO"

    runtime_dir: Path = Path("runtime")
    artifacts_dir: Path = Path("runtime/artifacts")
    queue_dir: Path = Path("runtime/queue")
    model_dir: Path = Path("runtime/models")

    page_detector_weights: Path | None = None
    detector_confidence: float = 0.35

    max_batch_workers: int = 4
    worker_poll_seconds: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OMR_", extra="ignore")


settings = Settings()

