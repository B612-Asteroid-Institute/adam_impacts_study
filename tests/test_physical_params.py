import json
import pytest

from unittest.mock import patch
from adam_impact_study.physical_params import select_albedo_from_range, select_asteroid_size, determine_ast_class, create_physical_params_single, PhotometricProperties


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


@patch("numpy.random.default_rng")
def test_determine_ast_class_returns_C(mock_rng):
    mock_rng.return_value.random.return_value = 0.1  # Simulating a random value less than percent_C
    percent_C = 0.5
    percent_S = 0.5
    assert determine_ast_class(percent_C, percent_S) == "C"


@patch("numpy.random.default_rng")
def test_determine_ast_class_returns_S(mock_rng):
    mock_rng.return_value.random.return_value = 0.9  # Simulating a random value greater than percent_C
    percent_C = 0.5
    percent_S = 0.5
    assert determine_ast_class(percent_C, percent_S) == "S"
    
    percent_C = 0.5
    percent_S = 0.5
    assert determine_ast_class(percent_C, percent_S) == "S"


def test_create_physical_params_single(tmp_path):
    config_data = {
        "C_albedo_min": 0.03,
        "C_albedo_max": 0.09,
        "S_albedo_min": 0.10,
        "S_albedo_max": 0.22,
        "percent_C": 1.0,
        "percent_S": 0.0,
        "min_diam": 0.001,
        "max_diam": 100,
        "n_asteroids": 1000,
        "u_r_C": 1.786,
        "g_r_C": 0.474,
        "i_r_C": -0.119,
        "z_r_C": -0.126,
        "y_r_C": -0.131,
        "u_r_S": 2.182,
        "g_r_S": 0.65,
        "i_r_S": -0.2,
        "z_r_S": -0.146,
        "y_r_S": -0.151,
}
    
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)

    obj_id = "test_obj_1"

    phys_params = create_physical_params_single(str(config_file), obj_id)

    assert isinstance(phys_params, PhotometricProperties)
    assert str(phys_params.ObjID[0]) == obj_id
    assert len(phys_params.H_mf) == 1
    assert len(phys_params.u_mf) == 1
    assert len(phys_params.g_mf) == 1
    assert len(phys_params.i_mf) == 1
    assert len(phys_params.z_mf) == 1
    assert len(phys_params.y_mf) == 1
    assert len(phys_params.GS) == 1
    assert phys_params.u_mf.to_numpy()[0] == 1.786
    assert phys_params.g_mf.to_numpy()[0] == 0.474
    assert phys_params.i_mf.to_numpy()[0] == -0.119
    assert phys_params.z_mf.to_numpy()[0] == -0.126
    assert phys_params.y_mf.to_numpy()[0] == -0.131
    assert phys_params.GS.to_numpy()[0] == 0.15
