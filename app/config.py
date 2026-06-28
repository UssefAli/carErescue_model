from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration for the OBD diagnostics microservice.

    This service is intentionally SEPARATE from the CarErescue backend:
    it has its own environment file, its own port, and its own settings.
    All values can be overridden via environment variables (Railway sets
    them in the dashboard) or the local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- service identity ---
    app_name: str = "CarErescue OBD Diagnostics"
    environment: str = "development"        # development | staging | production
    version: str = "0.1.0"

    # --- dataset ---
    # Folder (relative to the service root) holding the Toyota Etios CSVs.
    dataset_dir: str = "app/data/etios"
    default_vehicle_id: str = "etios-2014"

    # --- networking ---
    # Railway injects PORT; locally we default to 8001 so it never clashes
    # with the main CarErescue API on 8000.
    port: int = 8001
    cors_origins: List[str] = ["*"]

    @property
    def dataset_path(self) -> Path:
        root = Path(__file__).resolve().parent.parent
        return (root / self.dataset_dir).resolve()


settings = Settings()
