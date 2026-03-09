import pathlib

from hashi_puzzle_solver.models.config import HashiModelConfig
import pytest

CONFIGS_DIR = pathlib.Path(__file__).parent.parent / "configs"


def get_yaml_configs() -> list[pathlib.Path]:
    """
    Get all YAML config files in the configs directory.

    Returns
    -------
    list[pathlib.Path]
        List of paths to YAML config files.
    """
    # Filter out best_model.pt if it's accidentally picked up, though glob avoids it
    return [p for p in CONFIGS_DIR.glob("*.yaml") if p.is_file()]


@pytest.mark.parametrize("config_path", get_yaml_configs(), ids=lambda p: p.name)
def test_all_configs_loadable(config_path: pathlib.Path) -> None:
    """Ensure all YAML configs loadable into HashiModelConfig."""
    try:
        # This will fail if there are unexpected fields in the YAML
        # or if required fields are missing/wrong type
        config = HashiModelConfig.from_yaml(config_path)

        # Basic smoke test that data was actually loaded
        assert config.data is not None
        assert config.model is not None
        assert config.training is not None

    except TypeError as e:
        msg = f"Config {config_path.name} failed validation: {e}"
        pytest.fail(msg)
    except Exception as e:
        msg = f"Config {config_path.name} failed to load: {e}"
        pytest.fail(msg)


def test_model_config_has_all_legacy_fields() -> None:
    """Ensure ModelConfig has fields that might be used by legacy code."""
    from dataclasses import fields

    from hashi_puzzle_solver.models.config import ModelConfig

    field_names = {f.name for f in fields(ModelConfig)}
    assert "use_degree" in field_names


def test_training_config_has_all_expected_fields() -> None:
    """Ensure TrainingConfig has all recently added fields."""
    from dataclasses import fields

    from hashi_puzzle_solver.models.config import TrainingConfig

    field_names = {f.name for f in fields(TrainingConfig)}
    assert "weight_decay" in field_names
    assert "ar_max_steps" in field_names
    assert "gumbel_temperature" in field_names
    assert "model_dir" in field_names
