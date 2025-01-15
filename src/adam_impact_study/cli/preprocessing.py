import argparse
import logging
import os

from adam_impact_study.conversions import impactor_file_to_adam_orbit

logger = logging.getLogger(__name__)


def convert_impactor_file(
    input_file: str,
    output_file: str,
) -> None:
    """Convert impactor CSV file to ADAM orbit parquet format."""
    logger.info(f"Converting {input_file} to {output_file}")
    orbits = impactor_file_to_adam_orbit(input_file)
    orbits.to_parquet(output_file)
    logger.info(f"Converted {len(orbits)} orbits")


def main():
    """Main entry point for preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="Convert impactor files to ADAM format"
    )
    parser.add_argument("input_file", help="Input CSV file with impactor data")
    parser.add_argument("--output_file", help="Output parquet file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Re-use the same base for the filename as the input file, if not specified
    if args.output_file is None:
        args.output_file = os.path.splitext(args.input_file)[0] + ".parquet"

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    convert_impactor_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
