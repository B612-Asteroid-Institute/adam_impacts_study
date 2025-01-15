import os
import tempfile

import pytest

from adam_impact_study.population import PopulationConfig


@pytest.fixture
def default_population_config() -> PopulationConfig:
    return PopulationConfig.default()


def test_PopulationConfig_to_from_json(default_population_config: PopulationConfig):
    # Test that serialization to and from json works round-trip
    with tempfile.TemporaryDirectory() as temp_dir:
        json_file = os.path.join(temp_dir, "population_config.json")
        default_population_config.to_json(json_file)
        loaded_config = PopulationConfig.from_json(json_file)
        assert loaded_config == default_population_config
