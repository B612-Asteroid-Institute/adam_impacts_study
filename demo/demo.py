import logging
import os

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.conversions import impactor_file_to_adam_orbit
from adam_impact_study.impacts_study import run_impact_study_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the run name and directories
RUN_NAME = "Impact_Study_Demo"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
RUN_DIR = os.getcwd()
FO_DIR = os.path.join(os.path.dirname(__file__), "../find_orb/find_orb")

# Define the input files
impactors_file = os.path.join(os.path.dirname(__file__), "data/10_impactors.csv")
pointing_file = os.path.join(os.path.dirname(__file__), "data/baseline_v2.0_1yr.db")
chunk_size = 1

run_config_file = os.path.join(os.path.dirname(__file__), "impact_run_config.json")

impactor_orbits = impactor_file_to_adam_orbit(impactors_file)

impactor_orbits = impactor_orbits[0]

# Run the impact study
impact_study_results = run_impact_study_all(
    impactor_orbits,
    run_config_file,
    pointing_file,
    RUN_NAME,
    FO_DIR,
    RUN_DIR,
    RESULT_DIR,
    max_processes=1
)

logger.info(impact_study_results)

if impact_study_results is not None:
    plot_ip_over_time(impact_study_results)

