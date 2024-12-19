import argparse
import logging
import os

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.conversions import impactor_file_to_adam_orbit
from adam_impact_study.impacts_study import run_impact_study_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up command line arguments
parser = argparse.ArgumentParser(description='Run impact study demo')
parser.add_argument('--run-name', default="Impact_Study_Demo",
                   help='Name of the impact study run (default: Impact_Study_Demo)')
parser.add_argument('--result-dir', 
                   default=os.path.join(os.path.dirname(__file__), "results"),
                   help='Directory for storing results (default: ./results)')
parser.add_argument('--run-dir', default=os.getcwd(),
                   help='Working directory for the run (default: current working directory)')
parser.add_argument('--fo-dir', 
                   default=os.path.join(os.path.dirname(__file__), "find_orb/find_orb"),
                   help='Find_Orb directory path (default: ./find_orb/find_orb)')

args = parser.parse_args()

# Use the command line arguments or defaults
RUN_NAME = args.run_name
RESULT_DIR = args.result_dir
RUN_DIR = args.run_dir
FO_DIR = args.fo_dir

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

