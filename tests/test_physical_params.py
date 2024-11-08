from adam_impact_study.physical_params import select_albedo_from_range, select_asteroid_size, determine_ast_class
from unittest.mock import patch

def test_select_albedo_from_range():
    albedo_min = 0.1
    albedo_max = 1.0

    albedo = select_albedo_from_range(albedo_min, albedo_max)
    assert albedo >= albedo_min
    assert albedo <= albedo_max

def test_select_size():
    min_diam = 0.1
    max_diam = 1.0

    dist = select_asteroid_size(min_diam, max_diam)
    assert dist >= min_diam
    assert dist <= max_diam


@patch("adam_impact_study.physical_params.rng")
def test_determine_ast_class_returns_C(mock_rng):
    mock_rng.random.return_value = 0.1  # Assuming f_C > 0.1
    assert determine_ast_class() == "C"

@patch("adam_impact_study.physical_params.rng")
def test_determine_ast_class_returns_S(mock_rng):
    mock_rng.random.return_value = 0.9  # Assuming f_C < 0.9
    assert determine_ast_class() == "S"