import argparse
import logging
import os
from typing import Optional

from adam_core.orbits import Orbits

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.impacts_study import run_impact_study_all

logger = logging.getLogger(__name__)

def run_impact_study(
    orbit_file: str,
    run_dir: str,
    max_processes: int = 1,
    pointing_file: Optional[str] = None,
    population_config: Optional[str] = None,
    object_id: Optional[str] = None,
) -> None:
    """Run impact study on provided orbits."""
    # Load orbits directly from parquet
    logger.info(f"Loading orbits from {orbit_file}")
    impactor_orbits = Orbits.from_parquet(orbit_file)

    if object_id:
        impactor_orbits = impactor_orbits.select("object_id", object_id)
        logger.info(f"Filtered to single object: {object_id}")

    logger.info(f"Processing {len(impactor_orbits)} orbits")

    # Create output directory
    os.makedirs(run_dir, exist_ok=True)

    # Run impact study
    logger.info("Starting impact study...")
    impact_study_results = run_impact_study_all(
        impactor_orbits,
        population_config,
        pointing_file,
        run_dir,
        max_processes=max_processes,
    )

    logger.info("Generating plots...")
    plot_ip_over_time(impact_study_results, run_dir)
    logger.info(f"Results saved to {run_dir}")

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Run impact study")
    parser.add_argument("orbit_file", help="Path to orbit file (parquet format)")
    parser.add_argument("run_dir", help="Directory for this study run")
    parser.add_argument(
        "--max-processes", 
        type=int, 
        default=1,
        help="Maximum number of processes to use"
    )
    parser.add_argument(
        "--pointing-file",
        help="Path to pointing database file"
    )
    parser.add_argument(
        "--population-config",
        help="Path to population configuration file"
    )
    parser.add_argument(
        "--object-id",
        help="Specific object ID to process"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run_impact_study(
        args.orbit_file,
        args.run_dir,
        max_processes=args.max_processes,
        pointing_file=args.pointing_file,
        population_config=args.population_config,
        object_id=args.object_id,
    )

if __name__ == "__main__":
    main()
