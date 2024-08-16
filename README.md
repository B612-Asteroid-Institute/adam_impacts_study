# Adam Impact Study

## Description
This package includes all the functions and utilities needed to run the impact study.

## Installation

### Using PDM (recommended)
To install the necessary dependencies using PDM, run:

```bash
pdm install
```

This command will install all dependencies as specified in the `pyproject.toml` file.

### Using pip
If you do not have PDM installed, you can alternatively install dependencies using pip:

```bash
pip install -r requirements.txt
```

However, using PDM is recommended to ensure consistent and reproducible environments.

### Download Ephemeris Files

To download the assist ephemeris files, run:

```
python set_up_assist.py
```

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
