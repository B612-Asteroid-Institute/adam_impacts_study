import os

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.impacts_study import run_impact_study_all

# Define the run name and directories
RUN_NAME = "Impact_Study_Demo"
RESULT_DIR = "results"
RUN_DIR = os.getcwd()
FO_DIR = "../find_orb/find_orb"

# Define the input files
impactors_file = "data/10_impactors.csv"
pointing_file = "data/baseline_v2.0_1yr.db"
chunk_size = 1

physical_params_string = "15.88 1.72 0.48 -0.11 -0.12 -0.12 0.15"

# Run the impact study
impact_study_results = run_impact_study_all(
    impactors_file,
    physical_params_string,
    pointing_file,
    RUN_NAME,
    FO_DIR,
    RUN_DIR,
    RESULT_DIR,
    chunk_size,
)

print(impact_study_results)

if impact_study_results is not None:
    plot_ip_over_time(impact_study_results)
