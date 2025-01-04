import argparse
import logging
import os

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.conversions import impactor_file_to_adam_orbit
from adam_impact_study.impacts_study import run_impact_study_all

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up command line arguments
parser = argparse.ArgumentParser(description='Run impact study demo')
parser.add_argument('--run-name', default="Impact_Study_Demo",
                   help='Name of the impact study run (default: Impact_Study_Demo)')
parser.add_argument('--base-dir', default=os.getcwd(),
                   help='Base directory for all results (default: current working directory)')
parser.add_argument('--fo-dir', 
                   default=os.path.join(os.path.dirname(__file__), "../find_orb/find_orb"),
                   help='Find_Orb directory path (default: ../find_orb/find_orb)')
parser.add_argument('--max-processes', type=int, default=1,
                   help='Maximum number of processes to use for impact calculation (default: 1)')

args = parser.parse_args()

# Use the command line arguments or defaults
RUN_NAME = args.run_name
BASE_DIR = args.base_dir

# Define the input files
impactors_file = os.path.join(os.path.dirname(__file__), "data/10_impactors.csv")
pointing_file = os.path.join(os.path.dirname(__file__), "data/baseline_v2.0_1yr.db")
# pointing_file = os.path.join(os.path.dirname(__file__), "data/baseline_v3.6_10yrs.db")

population_config_file = os.path.join(os.path.dirname(__file__), "impact_run_config.json")

impactor_orbits = impactor_file_to_adam_orbit(impactors_file)

# impactor_orbits = impactor_orbits[0:20]
impactor_orbits = impactor_orbits.select("object_id", "I00007")


# Run the impact study
impact_study_results = run_impact_study_all(
    impactor_orbits,
    population_config_file,
    pointing_file,
    BASE_DIR,
    RUN_NAME,
    max_processes=args.max_processes
)

logger.info(impact_study_results)

if impact_study_results is not None:
    plot_ip_over_time(impact_study_results, BASE_DIR, RUN_NAME)

