import os
from unittest.mock import patch

from adam_core.orbits import Orbits

from adam_impact_study.fo_od import run_fo_od


@patch("subprocess.run")
def test_run_fo_od(mock_subprocess_run, tmpdir):
    fo_input_file = "input_file.txt"
    fo_output_folder = "output_folder"
    FO_DIR = tmpdir.mkdir("FO_DIR")
    RUN_DIR = tmpdir.mkdir("RUN_DIR")
    RESULT_DIR = tmpdir.mkdir("RESULT_DIR")

    # Create mock input files
    input_file_path = RESULT_DIR.join(fo_input_file)
    input_file_path.write("Dummy input content")

    # Create mock output files
    output_folder_path = FO_DIR.join(fo_output_folder)
    output_folder_path.mkdir()

    output_folder_path = RESULT_DIR.join(fo_output_folder)
    output_folder_path.mkdir()

    covar_file_path = FO_DIR.join(fo_output_folder).join("covar.json")
    covar_result_path = RESULT_DIR.join(fo_output_folder).join("covar.json")
    print(covar_file_path)
    covar_file_text = """{
        "covar": [
            [
                4.16121327342e-12,
                -2.09134780573e-12,
                -2.30283659349e-16,
                -2.71788684422e-14,
                -1.04967286688e-14,
                1.23426143706e-18
            ],
            [
                -2.09134780573e-12,
                1.06124575034e-12,
                3.56294827847e-17,
                1.35942552795e-14,
                5.18898046367e-15,
                6.94693054473e-19
            ],
            [
                -2.30283659349e-16,
                3.56294827847e-17,
                5.18082512922e-15,
                4.41513579676e-18,
                -1.11430144937e-18,
                -9.37068796782e-17
            ],
            [
                -2.71788684422e-14,
                1.35942552795e-14,
                4.41513579676e-18,
                2.08874755856e-16,
                3.6370326027e-17,
                -8.68973770791e-20
            ],
            [
                -1.04967286688e-14,
                5.18898046367e-15,
                -1.11430144937e-18,
                3.6370326027e-17,
                6.53394971075e-17,
                2.33681396804e-20
            ],
            [
                1.23426143706e-18,
                6.94693054473e-19,
                -9.37068796782e-17,
                -8.68973770791e-20,
                2.33681396804e-20,
                2.54650689185e-18
            ]
        ],
        "state_vect": [
            2.40431779633740117,
            -2.0586498601048886,
            1.56463294002342372e-05,
            0.00508148904172802708,
            -0.00632766941087369462,
            2.1947140448603267e-07
        ],
        "epoch": 2460490.842573
    }"""
    covar_file_path.write(covar_file_text)
    covar_result_path.write(covar_file_text)
    total_json_file_path = FO_DIR.join(fo_output_folder).join("total.json")
    total_json_result_path = RESULT_DIR.join(fo_output_folder).join("total.json")
    total_json_text = """{"num": 1,
  "ids":
  [
    "Test_1001"
  ],
  "objects":
  {
    "Test_1001":
    {
      "object": "Test_1001",
      "packed": "Test_1001",
      "created": 2460558.23098,
      "created iso": "2024-09-04T17:32:36Z",
      "Find_Orb_version": 2460533.50000,
      "Find_Orb_version_iso": "2024-08-11T00:00:00Z",
      "elements":
      {
        "central body": "Sun",
        "frame": "J2000 ecliptic",
        "reference": "Find_Orb",
        "epoch_iso": "2024-09-29T00:00:00Z",
        "epoch":  2460582.50000000,
        "P": 1394.9253892736097, "P sigma": 0.00253,
        "M":  77.3353733269255, "M sigma": 0.00015,
        "n":   0.2580783192909, "n sigma": 4.68e-7,
        "a":   2.4432525915365, "a sigma": 2.95e-6,
        "e":   0.9842422700762, "e sigma": 3.12e-7,
        "q":   0.0385001144730, "q sigma": 7.83e-7,
        "Q":   4.8480050685999, "Q sigma": 5.57e-6,
        "i": 179.9930186752411, "i sigma": 0.00006,
        "arg_per":  189.7896453405323, "arg_per sigma":  0.031,
        "asc_node": 321.7381294352738, "asc_node sigma": 0.031,
        "Tp": 2460282.84145627, "Tp sigma": 0.000342,
        "Tp_iso": "2023-12-04T08:11:41.821Z",
        "H":  16.06,
        "G":   0.15,
        "rms_residual": 0.033146,
        "weighted_rms_residual": 0.9631,
        "n_resids": 102,
        "U": 2.9896,
        "p_NEO": 100.0000,
        "MOIDs":
        {
          "Mercury" : 0.038916,
          "Venus" : 0.022080,
          "Earth" : 0.000009,
          "Mars" : 0.041731,
          "Jupiter" : 0.237747,
          "Saturn" : 5.066341,
          "Uranus" : 14.993002,
          "Neptune" : 25.227115
        }
      },
      "observations":
      {}
    }
    }
    }"""
    total_json_file_path.write(total_json_text)
    total_json_result_path.write(total_json_text)

    # Call the function under test
    result = run_fo_od(
        fo_input_file=str(fo_input_file),
        fo_output_folder=str(fo_output_folder),
        FO_DIR=str(FO_DIR),
        RUN_DIR=str(RUN_DIR),
        RESULT_DIR=str(RESULT_DIR),
    )

    assert os.path.exists(f"{FO_DIR}/{fo_input_file}")

    # Verify that correct find_orb command was run
    expected_command = (
        f"cd {FO_DIR}; ./fo {fo_input_file} "
        f"-O {fo_output_folder}; cp -r {fo_output_folder} "
        f"{RUN_DIR}/{RESULT_DIR}/; cd {RUN_DIR}"
    )
    mock_subprocess_run.assert_called_once_with(expected_command, shell=True)

    assert os.path.exists(f"{FO_DIR}/{fo_output_folder}/covar.json")

    assert isinstance(result, Orbits)
    assert result.orbit_id[0].as_py() == "Test_1001"
    assert result.coordinates.time.mjd()[0].as_py() - 60490.342573 < 1e-6
    assert result.coordinates.x[0].as_py() - 2.40431779633740117 < 1e-13
    assert result.coordinates.y[0].as_py() - -2.0586498601048886 < 1e-13
    assert result.coordinates.z[0].as_py() - 1.56463294002342372e-05 < 1e-13
    assert result.coordinates.vx[0].as_py() - 0.00508148904172802708 < 1e-13
    assert result.coordinates.vy[0].as_py() - -0.00632766941087369462 < 1e-13
    assert result.coordinates.vz[0].as_py() - 2.1947140448603267e-07 < 1e-13
    assert (
        result.coordinates.covariance.values[0][0].as_py() - 4.16121327342e-12 < 1e-13
    )
