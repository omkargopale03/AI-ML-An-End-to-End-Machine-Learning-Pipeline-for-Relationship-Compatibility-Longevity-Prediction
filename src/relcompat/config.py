"""Load pipeline configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_samples: int = 10_000
    missing_rate: float = 0.03
    outlier_rate: float = 0.005
    test_size: float = 0.2
    cv_folds: int = 5
    n_iter: int = 8
    n_jobs: int = -1
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @property
    def raw_path(self) -> Path:
        return self.data_dir / "raw" / "couples.csv"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    def with_overrides(self, **overrides) -> "Config":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path | None = None, **overrides) -> Config:
    """Read the YAML config; keyword overrides win over file values."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    data, training, paths = raw.get("data", {}), raw.get("training", {}), raw.get("paths", {})
    defaults = Config()
    cfg = Config(
        seed=raw.get("seed", defaults.seed),
        n_samples=data.get("n_samples", defaults.n_samples),
        missing_rate=data.get("missing_rate", defaults.missing_rate),
        outlier_rate=data.get("outlier_rate", defaults.outlier_rate),
        test_size=training.get("test_size", defaults.test_size),
        cv_folds=training.get("cv_folds", defaults.cv_folds),
        n_iter=training.get("n_iter", defaults.n_iter),
        n_jobs=training.get("n_jobs", defaults.n_jobs),
        data_dir=_resolve(paths.get("data_dir", defaults.data_dir)),
        models_dir=_resolve(paths.get("models_dir", defaults.models_dir)),
        reports_dir=_resolve(paths.get("reports_dir", defaults.reports_dir)),
    )
    return cfg.with_overrides(**overrides)
