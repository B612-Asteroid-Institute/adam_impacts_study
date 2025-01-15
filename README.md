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

You can run the impact study using the CLI, like this:

```bash
adam-impact demo/data/10_impactors.parquet demo/test_run/ --pointing-file demo/data/baseline_v2.0_1yr.db --population-config demo/data/population_config.json --object-id I00007 --debug
```

Or you can run the study in python, to dynamically control the parameters.

This code can be used to run the impact study, examining impact probabilities over time. For example, the demo code below can be run to look at the expected impact probabilities of 10 synthetic objects over the course of one year.

```
import os

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.impacts_study import run_impact_study_fo

# Define the run name and directories
RUN_NAME = "Impact_Study_Demo"
RUN_DIR = os.getcwd()

# Define the input files
impactors_file = "data/10_impactors.parquet"
pointing_file = "data/baseline_v2.0_1yr.db"
population_config = "data/population_config.json"


# Run the impact study
impact_study_results = run_impact_study_all(
    impactor_orbits,
    population_config,
    pointing_file,
    run_dir,
    max_processes=max_processes,
)

logger.info(impact_study_results)

plot_ip_over_time(impact_study_results)
```

This code and the input files needed to run it can be found in the 'demo' section.
