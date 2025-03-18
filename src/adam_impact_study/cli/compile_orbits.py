import glob
import os

import pyarrow as pa
import quivr as qv

from adam_impact_study.types import ImpactorOrbits


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Combine orbit files from a directory")
    parser.add_argument("input_dir", help="Directory containing orbit parquet files")
    parser.add_argument("output_file", help="Output parquet file path")
    parser.add_argument(
        "--num-orbits",
        type=int,
        default=None,
        required=False,
        help="Number of orbits to take from each file (default: all)",
    )
    args = parser.parse_args()

    # Get list of all parquet files in the input directory
    parquet_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))

    # Initialize empty list to store combined orbits
    all_orbits = []

    for parquet_file in parquet_files:
        # Extract year from filename
        filename = os.path.basename(parquet_file)
        year = filename.split("_")[1]
        print(f"Processing {parquet_file} ({year})")

        # Load orbits from parquet file
        orbits = ImpactorOrbits.from_parquet(parquet_file)

        # Take specified number of orbits
        if args.num_orbits is not None and len(orbits) > args.num_orbits:
            orbits = orbits[: args.num_orbits]

        # Add year suffix to object IDs
        new_ids = [f"{id.as_py()}_{year}" for id in orbits.orbit_id]
        ids_pa = pa.array(new_ids, type=pa.large_string())

        orbits = orbits.set_column("orbit_id", ids_pa)

        all_orbits.append(orbits)

    # Combine all orbit lists
    combined_orbits = qv.concatenate(all_orbits)

    # Save combined orbits to parquet
    combined_orbits.to_parquet(args.output_file)
    print(f"Saved {len(combined_orbits)} combined orbits to {args.output_file}")


if __name__ == "__main__":
    main()
