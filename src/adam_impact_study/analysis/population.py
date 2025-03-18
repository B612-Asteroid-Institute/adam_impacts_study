import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import quivr as qv

from ..types import ImpactorOrbits, ImpactorResultSummary


class OrbitalElementRecoveryStatistics(qv.Table):
    a_bin_min = qv.Float64Column()
    a_bin_max = qv.Float64Column()
    i_bin_min = qv.Float64Column()
    i_bin_max = qv.Float64Column()
    e_bin_min = qv.Float64Column()
    e_bin_max = qv.Float64Column()
    recovered = qv.Int64Column()
    total = qv.Int64Column()


def compute_orbital_element_recovery_statistics(
    impactor_orbits: ImpactorOrbits,
    results: ImpactorResultSummary,
):
    """
    Generate statistics about recovered vs initial population
    """
    recovered_orbits = results.apply_mask(results.discovered())
    input_keplerian_elements = impactor_orbits.coordinates.to_keplerian()
    recovered_keplerian_elements = recovered_orbits.orbit.coordinates.to_keplerian()
    # Create bins of a vs i and a vs e using min and max of the input orbits
    a_min = np.floor(np.min(input_keplerian_elements.a))
    a_max = np.ceil(np.max(input_keplerian_elements.a))
    i_min = np.floor(np.min(input_keplerian_elements.i))
    i_max = np.ceil(np.max(input_keplerian_elements.i))
    e_min = np.floor(np.min(input_keplerian_elements.e))
    e_max = np.ceil(np.max(input_keplerian_elements.e))
    # Use range to let us set the bin widths
    num_a_bins = 10
    num_i_bins = 10
    num_e_bins = 10
    a_bins = np.linspace(a_min, a_max, num_a_bins)
    i_bins = np.linspace(i_min, i_max, num_i_bins)
    e_bins = np.linspace(e_min, e_max, num_e_bins)

    a_bin_width = a_bins[1] - a_bins[0]
    i_bin_width = i_bins[1] - i_bins[0]
    e_bin_width = e_bins[1] - e_bins[0]

    # Create a grid representing all a, e, and i bin values
    a_grid, i_grid, e_grid = np.meshgrid(a_bins, i_bins, e_bins)
    a_grid = a_grid.flatten()
    i_grid = i_grid.flatten()
    e_grid = e_grid.flatten()

    statistics = OrbitalElementRecoveryStatistics.empty()

    for a, i, e in zip(a_grid, i_grid, e_grid):
        a_bin_min = a - a_bin_width / 2.0
        a_bin_max = a + a_bin_width / 2.0
        i_bin_min = i - i_bin_width / 2.0
        i_bin_max = i + i_bin_width / 2.0
        e_bin_min = e - e_bin_width / 2.0
        e_bin_max = e + e_bin_width / 2.0

        a_input_mask = pc.and_(
            pc.greater_equal(input_keplerian_elements.a, a_bin_min),
            pc.less(input_keplerian_elements.a, a_bin_max),
        )
        i_input_mask = pc.and_(
            pc.greater_equal(input_keplerian_elements.i, i_bin_min),
            pc.less(input_keplerian_elements.i, i_bin_max),
        )
        e_input_mask = pc.and_(
            pc.greater_equal(input_keplerian_elements.e, e_bin_min),
            pc.less(input_keplerian_elements.e, e_bin_max),
        )

        a_recovered_mask = pc.and_(
            pc.greater_equal(recovered_keplerian_elements.a, a_bin_min),
            pc.less(recovered_keplerian_elements.a, a_bin_max),
        )

        i_recovered_mask = pc.and_(
            pc.greater_equal(recovered_keplerian_elements.i, i_bin_min),
            pc.less(recovered_keplerian_elements.i, i_bin_max),
        )
        e_recovered_mask = pc.and_(
            pc.greater_equal(recovered_keplerian_elements.e, e_bin_min),
            pc.less(recovered_keplerian_elements.e, e_bin_max),
        )

        combined_input_mask = pc.and_(pc.and_(a_input_mask, i_input_mask), e_input_mask)
        combined_recovered_mask = pc.and_(
            pc.and_(a_recovered_mask, i_recovered_mask), e_recovered_mask
        )

        input_counts = len(input_keplerian_elements.apply_mask(combined_input_mask))
        recovered_counts = len(
            recovered_keplerian_elements.apply_mask(combined_recovered_mask)
        )

        statistics = qv.concatenate(
            [
                statistics,
                OrbitalElementRecoveryStatistics.from_kwargs(
                    a_bin_min=[a_bin_min],
                    a_bin_max=[a_bin_max],
                    i_bin_min=[i_bin_min],
                    i_bin_max=[i_bin_max],
                    e_bin_min=[e_bin_min],
                    e_bin_max=[e_bin_max],
                    recovered=[recovered_counts],
                    total=[input_counts],
                ),
            ]
        )

    return statistics


def plot_orbital_element_recovery_statistics(
    statistics: OrbitalElementRecoveryStatistics,
):
    # First we want to group by pairs of a and i and pairs of a and e
    grouped_by_a_e = statistics.table.group_by(
        ["a_bin_min", "a_bin_max", "e_bin_min", "e_bin_max"]
    ).aggregate([("recovered", "sum"), ("total", "sum")])
    grouped_by_a_i = statistics.table.group_by(
        ["a_bin_min", "a_bin_max", "i_bin_min", "i_bin_max"]
    ).aggregate([("recovered", "sum"), ("total", "sum")])

    # Here we want to conver to a 2-D array
    a_e_recovery_rates = grouped_by_a_e.to_pandas()
    a_i_recovery_rates = grouped_by_a_i.to_pandas()

    # Calculate the overall min and max for 'a' to ensure consistent x-axis
    a_min = min(
        a_e_recovery_rates["a_bin_min"].min(), a_i_recovery_rates["a_bin_min"].min()
    )
    a_max = max(
        a_e_recovery_rates["a_bin_max"].max(), a_i_recovery_rates["a_bin_max"].max()
    )

    a_samples = np.linspace(0.49, a_max, 100)
    np.linspace(
        np.min(grouped_by_a_e.column("e_bin_min").to_numpy()),
        np.max(grouped_by_a_e.column("e_bin_max").to_numpy()),
        100,
    )

    perihelion_line_neo = 1 - (1.3 / a_samples)

    # Create figure with adjusted layout for colorbar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Adjust bottom margin to prevent label overlap
    plt.subplots_adjust(bottom=0.25)  # Increased from 0.2 to 0.25

    # Create scatter plots with same normalization
    norm = plt.Normalize(vmin=0, vmax=1)

    sc1 = ax1.scatter(
        a_e_recovery_rates["a_bin_min"],
        a_e_recovery_rates["e_bin_min"],
        c=a_e_recovery_rates["recovered_sum"] / a_e_recovery_rates["total_sum"],
        norm=norm,
    )

    ax1.set_xlabel("Semi-major Axis (au)")
    ax1.set_ylabel("Eccentricity")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(a_min, a_max)
    ax1.plot(a_samples, perihelion_line_neo, "k--")

    ax2.scatter(
        a_i_recovery_rates["a_bin_min"],
        a_i_recovery_rates["i_bin_min"],
        c=a_i_recovery_rates["recovered_sum"] / a_i_recovery_rates["total_sum"],
        norm=norm,
    )

    ax2.set_xlabel("Semi-major Axis (au)")
    ax2.set_ylabel("Inclination (deg)")
    ax2.set_xlim(a_min, a_max)

    # Add single colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.04])  # Adjusted position slightly higher
    fig.colorbar(sc1, cax=cbar_ax, orientation="horizontal", label="Recovery Rate")

    plt.show()
