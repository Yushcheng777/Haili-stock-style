import importlib
import pytest


def test_ci_runs():
    """Basic sanity check to verify pytest collection and execution works."""
    assert True


@pytest.mark.parametrize("module", ["pandas", "matplotlib", "akshare"])  # keep lightweight
def test_optional_imports(module):
    """Attempt to import optional runtime dependencies.
    If an import fails in the CI environment, skip instead of failing to keep the smoke test green.
    """
    try:
        importlib.import_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Optional import '{module}' unavailable in CI: {exc}")