# Adam Impact Study

## Description
This package includes all the functions and utilities needed to run the impact study.

## Installation
First clone the repository with git.
### Using PDM (recommended)
To install the necessary dependencies using PDM, run:

```bash
pdm install
```

This command will install all dependencies as specified in the `pyproject.toml` file.

### Using pip
If you do not have PDM installed, you can alternatively install dependencies using pip:

```bash
pip install .
```

However, using PDM is recommended to ensure consistent and reproducible environments.

You will also need to decide which OD platform you want to use. Currently only Find Orb is available.

### Install Find Orb

Find orb is an orbit determination package.

To install find_orb, run the following script:

```
sh build_fo.sh
```

## Development Setup

If you are planning to contribute to this project or work on it in a development environment, you should install additional development dependencies. This can be done by running:

```bash
pdm install -G dev
```

If you are not using PDM and want to install the development dependencies with pip, you can run:

```bash
pip install .[dev]
```

## Usage
=======

This code can be used to run the impact study, examining impact probabilities over time. For example, the demo code below can be run to look at the expected impact probabilities of 10 synthetic objects over the course of one year.

```
import os

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.impacts_study import run_impact_study_fo

# Define the run name and directories
RUN_NAME = "Impact_Study_Demo"
RESULT_DIR = "results"
RUN_DIR = os.getcwd()
FO_DIR = "../find_orb/find_orb"

# Define the input files
impactors_file = "data/10_impactors.csv"
pointing_file = "data/baseline_v2.0_1yr.db"
sorcha_config_file = "data/sorcha_config_demo.ini"

# Additional file names generated from the run name
sorcha_orbits_file = f"data/sorcha_input_{RUN_NAME}.csv"
sorcha_physical_params_file = f"data/sorcha_params_{RUN_NAME}.csv"
sorcha_output_name = f"sorcha_output_{RUN_NAME}"
sorcha_output_file = f"{sorcha_output_name}.csv"
fo_input_file_base = f"fo_input_{RUN_NAME}"
fo_output_file_base = f"fo_output_{RUN_NAME}"

physical_params_string = "15.88 1.72 0.48 -0.11 -0.12 -0.12 0.15"

# Run the impact study
impact_study_results = run_impact_study_fo(
    impactors_file,
    sorcha_config_file,
    sorcha_orbits_file,
    sorcha_physical_params_file,
    sorcha_output_file,
    physical_params_string,
    pointing_file,
    sorcha_output_name,
    fo_input_file_base,
    fo_output_file_base,
    FO_DIR,
    RUN_DIR,
    RESULT_DIR,
)

logger.info(impact_study_results)

plot_ip_over_time(impact_study_results)
```

This code and the input files needed to run it can be found in the 'demo' section.
