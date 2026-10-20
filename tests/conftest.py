import pytest

from relcompat.config import load_config
from relcompat.data.generate import generate_couples
from relcompat.pipeline import run_pipeline


@pytest.fixture(scope="session")
def small_df():
    return generate_couples(n=800, seed=7)


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    """Train a tiny version of the full pipeline into a temp directory."""
    root = tmp_path_factory.mktemp("run")
    cfg = load_config(
        n_samples=600, cv_folds=2, n_iter=1, n_jobs=1,
        data_dir=root / "data", models_dir=root / "models", reports_dir=root / "reports",
    )
    run_pipeline(cfg, make_plots=False)
    return cfg.models_dir
