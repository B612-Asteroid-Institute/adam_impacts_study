import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import quivr as qv
import ray
import seaborn as sns
from adam_assist import ASSISTPropagator
from adam_core.constants import KM_P_AU
from adam_core.orbits import Orbits
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp


@pytest.fixture
def non_impacting_orbit_fixture():
    return non_impacting_orbit()


def non_impacting_orbits():
    """Load all non-impacting orbits from the round trip data."""
    orbit_path = os.path.join(
        os.path.dirname(__file__),
        "../data/inputs/12-16-2024--round-trip/ImpactorsStudy_2125-05-05T00_00_00_2135-05-06T00_00_00_non_impacting_objects.parquet",
    )
    return Orbits.from_parquet(orbit_path)


def non_impacting_orbit():
    """Load a single non-impacting orbit from the round trip data."""
    orbits = non_impacting_orbits()
    return orbits[1]


def impacting_orbits():
    """Load all impacting orbits from the round trip data."""
    orbit_path = os.path.join(
        os.path.dirname(__file__),
        "../data/inputs/12-16-2024--round-trip/ImpactorsStudy_2125-05-05T00_00_00_2135-05-06T00_00_00_impacting_objects.parquet",
    )
    return Orbits.from_parquet(orbit_path)


def impacting_orbit():
    """Load a single impacting orbit from the round trip data."""
    orbits = impacting_orbits()
    # Take just the first orbit
    return orbits[0:1]


@pytest.fixture
def impacting_orbit_fixture():
    return impacting_orbit()


class RoundTripResults(qv.Table):
    orbit_id = qv.LargeStringColumn(nullable=True)
    initial_dt = qv.Float64Column(nullable=True)
    min_dt = qv.Float64Column(nullable=True)
    adaptive_mode = qv.Int64Column(nullable=True)
    position_diff_km = qv.Float64Column(nullable=True)
    impact_mjd = qv.Float64Column(nullable=True)
    forward_steps_done = qv.Int64Column(nullable=True)
    backward_steps_done = qv.Int64Column(nullable=True)
    time_taken = qv.Float64Column(nullable=True)
    epsilon = qv.Float64Column(nullable=True)


def round_trip_worker(
    orbit: Orbits, min_dt: float, initial_dt: float, adaptive_mode: int, epsilon: float
):
    start = time.time()
    propagator = ASSISTPropagator(
        min_dt=min_dt,
        initial_dt=initial_dt,
        adaptive_mode=adaptive_mode,
        epsilon=epsilon,
    )
    time_2025 = Timestamp.from_iso8601("2025-05-05T00:00:00")
    original_epoch = orbit.coordinates.time
    forward_orbit = propagator.propagate_orbits(orbit, time_2025)
    forward_steps_done = propagator._last_simulation.steps_done
    # back_orbit = propagator.propagate_orbits(forward_orbit, original_epoch.add_days(-60))
    back_orbit = propagator.propagate_orbits(forward_orbit, original_epoch)
    backward_steps_done = propagator._last_simulation.steps_done
    diff = orbit.coordinates.r - back_orbit.coordinates.r
    norm_diff = np.linalg.norm(diff)
    diff_km = norm_diff * KM_P_AU

    # Check for impacts
    impact_results, impacts = propagator.detect_impacts(back_orbit, 60)
    impact_mjd = None
    if len(impacts) > 0:
        impact_mjd = impacts.coordinates.time.mjd()[0].as_py()
    end = time.time()
    print(
        f"end {orbit.orbit_id[0].as_py()} | {initial_dt} | {min_dt} | {adaptive_mode} | {epsilon} | {diff_km} | {impact_mjd} | {forward_steps_done} | {backward_steps_done} | {end - start}"
    )

    return RoundTripResults.from_kwargs(
        orbit_id=orbit.orbit_id,
        initial_dt=[initial_dt],
        min_dt=[min_dt],
        adaptive_mode=[adaptive_mode],
        position_diff_km=[diff_km],
        impact_mjd=[impact_mjd],
        forward_steps_done=[forward_steps_done],
        backward_steps_done=[backward_steps_done],
        time_taken=[end - start],
        epsilon=[epsilon],
    )


round_trip_worker_remote = ray.remote(round_trip_worker)


def test_round_trip_propagation_non_impacting(orbits: Orbits, max_processes: int = 1):
    """Run round trip propagation tests with various parameters and collect results."""

    min_dts = np.logspace(0, -5, 6)
    initial_dts = np.logspace(-3, -5, 3)
    adaptive_modes = [0, 1, 2, 3]
    # epsilons = np.logspace(-6, -9, 4)
    epsilons = np.logspace(-6, -7, 2)

    initialize_use_ray(num_cpus=max_processes)
    futures = []
    all_results = RoundTripResults.empty()
    for epsilon in epsilons:
        for orbit in orbits:
            for initial_dt in initial_dts:
                for min_dt in min_dts:
                    for adaptive_mode in adaptive_modes:
                        if min_dt > initial_dt:
                            continue

                        if max_processes == 1:
                            all_results = qv.concatenate(
                                [
                                    all_results,
                                    round_trip_worker(
                                        orbit, min_dt, initial_dt, adaptive_mode, epsilon
                                    ),
                                ]
                            )
                        else:
                            futures.append(
                                round_trip_worker_remote.remote(
                                    orbit, min_dt, initial_dt, adaptive_mode, epsilon
                                )
                            )

                        if len(futures) > max_processes * 1.5:
                            finished, futures = ray.wait(futures, num_returns=1)
                            result = ray.get(finished[0])
                            all_results = qv.concatenate([all_results, result])

    while len(futures) > 0:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        all_results = qv.concatenate([all_results, result])

    return all_results


def analyze_round_trip_results(results: RoundTripResults):
    """Analyze the results of round trip propagation tests.
    
    Args:
        results: RoundTripResults table containing the test results
        
    Returns:
        tuple: (correlation_matrix, best_configs, df) where:
            - correlation_matrix is a pandas DataFrame of correlations
            - best_configs is a pandas DataFrame of optimal configurations
            - df is the processed pandas DataFrame for further analysis
    """
    # Convert to pandas for easier analysis
    df = results.to_pandas()
    df['total_steps'] = df['forward_steps_done'] + df['backward_steps_done']
    
    # Correlation analysis
    correlation_vars = ['initial_dt', 'min_dt', 'epsilon', 'adaptive_mode', 
                       'position_diff_km', 'total_steps', 'time_taken']
    correlation_matrix = df[correlation_vars].corr()
    
    # Group by parameters and calculate mean metrics
    grouped = df.groupby(['initial_dt', 'min_dt', 'epsilon', 'adaptive_mode']).agg({
        'position_diff_km': 'mean',
        'total_steps': 'mean',
        'time_taken': 'mean',
        'impact_mjd': lambda x: x.notna().mean()
    }).reset_index()
    
    # Find best combinations
    best_configs = grouped[
        grouped['impact_mjd'] > 0.99
    ].sort_values(['total_steps', 'time_taken'])
    
    print("\nCorrelations with position difference (km):")
    print(correlation_matrix['position_diff_km'].sort_values(ascending=False))
    
    print("\nTop 5 most efficient configurations that reliably detect impacts:")
    print(best_configs[['initial_dt', 'min_dt', 'epsilon', 'adaptive_mode', 
                       'total_steps', 'time_taken', 'impact_mjd']].head())
    
    return correlation_matrix, best_configs, df


def visualize_round_trip_results(df: pd.DataFrame, save_path: str = None):
    """Create visualizations for round trip propagation results.
    
    Args:
        df: Processed pandas DataFrame from analyze_round_trip_results
        save_path: Optional path to save plots. If None, plots will be displayed.
    """
    # Correlation heatmap
    correlation_vars = ['initial_dt', 'min_dt', 'epsilon', 'adaptive_mode', 
                       'position_diff_km', 'total_steps', 'time_taken']
    correlation_matrix = df[correlation_vars].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Parameters')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/correlation_heatmap.png")
    else:
        plt.show()
    plt.close()

    # Pairplot for key parameters
    key_vars = ['initial_dt', 'min_dt', 'epsilon', 'adaptive_mode', 'position_diff_km']
    pairplot = sns.pairplot(df[key_vars], diag_kind='kde', plot_kws={'alpha': 0.6})
    pairplot.fig.suptitle('Parameter Relationships', y=1.02)
    if save_path:
        pairplot.savefig(f"{save_path}/parameter_relationships.png")
    else:
        plt.show()
    plt.close()

    # Parameter impact plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Effects on Position Difference')
    
    sns.boxplot(x='adaptive_mode', y='position_diff_km', data=df, ax=axes[0,0])
    axes[0,0].set_title('Adaptive Mode vs Position Difference')
    
    sns.scatterplot(x='initial_dt', y='position_diff_km', data=df, ax=axes[0,1])
    axes[0,1].set_xscale('log')
    axes[0,1].set_title('Initial dt vs Position Difference')
    
    sns.scatterplot(x='min_dt', y='position_diff_km', data=df, ax=axes[1,0])
    axes[1,0].set_xscale('log')
    axes[1,0].set_title('Min dt vs Position Difference')
    
    sns.scatterplot(x='epsilon', y='position_diff_km', data=df, ax=axes[1,1])
    axes[1,1].set_xscale('log')
    axes[1,1].set_title('Epsilon vs Position Difference')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/parameter_effects.png")
    else:
        plt.show()
    plt.close()

# Usage example:
# results = test_round_trip_propagation_non_impacting(orbits, max_processes=4)
# correlations, best_configs, df = analyze_round_trip_results(results)
# visualize_round_trip_results(df, save_path="path_to_save_plots")
