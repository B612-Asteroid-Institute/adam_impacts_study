import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import CartesianCoordinates, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Orbits, VariantOrbits
from adam_core.time import Timestamp


class Photometry(qv.Table):
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn()


class Observations(qv.Table):
    obs_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()
    photometry = Photometry.as_column(nullable=True)
    observing_night = qv.Int64Column(nullable=True)
    #: Was this observation linked by Rubin's SSP
    linked = qv.BooleanColumn(nullable=True)


class PhotometricProperties(qv.Table):
    orbit_id = qv.LargeStringColumn()
    H_mf = qv.Float64Column()
    u_mf = qv.Float64Column()
    g_mf = qv.Float64Column()
    i_mf = qv.Float64Column()
    z_mf = qv.Float64Column()
    y_mf = qv.Float64Column()
    GS = qv.Float64Column()


class ImpactorOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn()
    coordinates = CartesianCoordinates.as_column()
    impact_time = Timestamp.as_column()
    dynamical_class = qv.LargeStringColumn()
    ast_class = qv.LargeStringColumn()
    diameter = qv.Float64Column()  # km
    albedo = qv.Float64Column()
    H_r = qv.Float64Column()
    u_r = qv.Float64Column()
    g_r = qv.Float64Column()
    i_r = qv.Float64Column()
    z_r = qv.Float64Column()
    y_r = qv.Float64Column()
    GS = qv.Float64Column()

    def orbits(self) -> Orbits:
        return Orbits.from_kwargs(
            orbit_id=self.orbit_id,
            object_id=self.object_id,
            coordinates=self.coordinates,
        )

    def photometric_properties(self) -> PhotometricProperties:
        return PhotometricProperties.from_kwargs(
            orbit_id=self.orbit_id,
            H_mf=self.H_r,
            u_mf=self.u_r,
            g_mf=self.g_r,
            i_mf=self.i_r,
            z_mf=self.z_r,
            y_mf=self.y_r,
            GS=self.GS,
        )


class WindowResult(qv.Table):
    """
    Impact results from a select set of observations for a single orbit

    This is the result of fitting an orbit to a subset of observations and
    then running the impact probability calculation on the orbit.
    """

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    window = qv.LargeStringColumn()
    condition_id = qv.LargeStringColumn(
        nullable=True
    )  # nullable for backwards compatibility
    status = qv.LargeStringColumn(
        default="incomplete"
    )  # "complete", "incomplete", "failed"
    observation_start = Timestamp.as_column(nullable=True)
    observation_end = Timestamp.as_column(nullable=True)
    observation_count = qv.UInt64Column(nullable=True)
    observations_rejected = qv.UInt64Column(nullable=True)
    observation_nights = qv.UInt64Column(nullable=True)
    impact_probability = qv.Float64Column(nullable=True)
    mean_impact_time = Timestamp.as_column(nullable=True)
    minimum_impact_time = Timestamp.as_column(nullable=True)
    maximum_impact_time = Timestamp.as_column(nullable=True)
    stddev_impact_time = qv.Float64Column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)
    od_runtime = qv.Float64Column(nullable=True)
    ip_runtime = qv.Float64Column(nullable=True)
    window_runtime = qv.Float64Column(nullable=True)

    def complete(self) -> pa.BooleanArray:
        return pc.equal(self.status, "complete")

    def incomplete(self) -> pa.BooleanArray:
        return pc.equal(self.status, "incomplete")

    def failed(self) -> pa.BooleanArray:
        return pc.equal(self.status, "failed")

    def arc_length(self) -> pa.FloatArray:
        """
        Return the window arc length in days
        """
        return pc.subtract(self.observation_end.mjd(), self.observation_start.mjd())


class ResultsTiming(qv.Table):
    orbit_id = qv.LargeStringColumn()
    sorcha_runtime = qv.Float64Column(nullable=True)
    mean_od_runtime = qv.Float64Column(nullable=True)
    total_od_runtime = qv.Float64Column(nullable=True)
    mean_ip_runtime = qv.Float64Column(nullable=True)
    total_ip_runtime = qv.Float64Column(nullable=True)
    mean_window_runtime = qv.Float64Column(nullable=True)
    total_window_runtime = qv.Float64Column(nullable=True)
    total_runtime = qv.Float64Column(nullable=True)


# Define Earth's approximate orbital period in days
EARTH_ORBITAL_PERIOD_DAYS = 365.256363004


class ImpactorResultSummary(qv.Table):
    orbit = ImpactorOrbits.as_column()
    # This is a mean of means of the impact time from each window
    mean_impact_time = Timestamp.as_column(nullable=True)
    # Number of distinct orbit fitting windows
    windows = qv.UInt64Column()
    # Number of distinct nights of observations
    nights = qv.UInt64Column()
    # Number of distinct observations recovered
    observations = qv.UInt64Column()
    # Number of singletons from observations recovered
    singletons = qv.UInt64Column()
    # Number of tracklets from observations recovered
    tracklets = qv.UInt64Column()
    # Date of first observation
    first_observation = Timestamp.as_column(nullable=True)
    # Date of last observation
    last_observation = Timestamp.as_column(nullable=True)
    # Time when observations met minimum artificial discovery criteria
    # Currently set to 3 unique nights of tracklets
    discovery_time = Timestamp.as_column(nullable=True)
    # Time when observations met optimistic discovery criteria
    # Currently set to 6 observations in 30 days
    discovery_time_optimistic = Timestamp.as_column(nullable=True)
    # Impact probability at discovery time
    ip_at_discovery_time = qv.Float64Column(nullable=True)
    # Date object first reaches 0.01% impact probability
    ip_threshold_0_dot_01_percent = Timestamp.as_column(nullable=True)
    # Date object first reaches 1% impact probability
    ip_threshold_1_percent = Timestamp.as_column(nullable=True)
    # Date object first reaches 10% impact probability
    ip_threshold_10_percent = Timestamp.as_column(nullable=True)
    # Date object first reaches 50% impact probability
    ip_threshold_50_percent = Timestamp.as_column(nullable=True)
    # Date object first reaches 90% impact probability
    ip_threshold_90_percent = Timestamp.as_column(nullable=True)
    # Date object first reaches 99% impact probability
    ip_threshold_100_percent = Timestamp.as_column(nullable=True)
    # How close all the windows got to discovering the definite impact nature
    maximum_impact_probability = qv.Float64Column(nullable=True)
    # Error message from the impact study
    error = qv.LargeStringColumn(nullable=True)
    # Runtime of the impact study
    results_timing = ResultsTiming.as_column(nullable=True)
    #: Processing status of this particular orbit
    status = qv.LargeStringColumn(nullable=True, default="incomplete")

    def complete(self) -> pa.BooleanArray:
        return pc.equal(self.status, "complete")

    def incomplete(self) -> pa.BooleanArray:
        return pc.equal(self.status, "incomplete")

    def failed(self) -> pa.BooleanArray:
        return pc.equal(self.status, "failed")

    def discovered(self) -> pa.BooleanArray:
        return pc.and_(
            pc.invert(pc.is_null(self.discovery_time.days)),
            self.complete(),
        )

    def observed_but_not_discovered(self) -> pa.BooleanArray:
        return pc.and_(
            self.complete(),
            pc.and_(
                pc.is_null(self.discovery_time.days),
                pc.greater(self.observations, 0),
            ),
        )

    def summarize_discoveries(self) -> "DiscoverySummary":
        # Filter to completed orbits
        completed_orbits = self.apply_mask(self.complete())

        summary_table = completed_orbits.flattened_table().append_column(
            "discovered", completed_orbits.discovered()
        )
        discoveries_by_diameter_class = summary_table.group_by(
            ["orbit.diameter", "orbit.ast_class"]
        ).aggregate([("discovered", "sum"), ("discovered", "count")])
        discoveries_by_diameter_class = discoveries_by_diameter_class.append_column(
            "percent_discovered",
            pc.multiply(
                pc.divide(
                    pc.cast(
                        discoveries_by_diameter_class["discovered_sum"], pa.float64()
                    ),
                    pc.cast(
                        discoveries_by_diameter_class["discovered_count"], pa.float64()
                    ),
                ),
                100,
            ),
        ).sort_by([("orbit.diameter", "ascending")])

        return DiscoverySummary.from_kwargs(
            diameter=discoveries_by_diameter_class["orbit.diameter"],
            ast_class=discoveries_by_diameter_class["orbit.ast_class"],
            discovered=discoveries_by_diameter_class["discovered_sum"],
            total=discoveries_by_diameter_class["discovered_count"],
        )

    def warning_time(self) -> pa.FloatArray:
        """
        Time in days from impact time to 1% IP threshold or discovery time
        (whichever is later)

        This method does not distinguish between discovered and not discovered.
        """
        return pc.max_element_wise(
            pc.subtract(
                self.orbit.impact_time.mjd(), self.ip_threshold_1_percent.mjd()
            ),
            pc.subtract(self.orbit.impact_time.mjd(), self.discovery_time.mjd()),
        )

    def arc_length(self) -> pa.FloatArray:
        return pc.subtract(self.last_observation.mjd(), self.first_observation.mjd())

    def days_discovery_to_0_dot_01_percent(self) -> pa.FloatArray:
        return pc.subtract(
            self.ip_threshold_0_dot_01_percent.mjd(), self.discovery_time.mjd()
        )

    def days_discovery_to_1_percent(self) -> pa.FloatArray:
        return pc.subtract(self.ip_threshold_1_percent.mjd(), self.discovery_time.mjd())

    def days_discovery_to_10_percent(self) -> pa.FloatArray:
        return pc.subtract(
            self.ip_threshold_10_percent.mjd(), self.discovery_time.mjd()
        )

    def days_discovery_to_50_percent(self) -> pa.FloatArray:
        return pc.subtract(
            self.ip_threshold_50_percent.mjd(), self.discovery_time.mjd()
        )

    def days_discovery_to_90_percent(self) -> pa.FloatArray:
        return pc.subtract(
            self.ip_threshold_90_percent.mjd(), self.discovery_time.mjd()
        )

    def days_discovery_to_100_percent(self) -> pa.FloatArray:
        return pc.subtract(
            self.ip_threshold_100_percent.mjd(), self.discovery_time.mjd()
        )

    def days_0_dot_01_percent_to_impact(self) -> pa.FloatArray:
        return pc.subtract(
            self.orbit.impact_time.mjd(), self.ip_threshold_0_dot_01_percent.mjd()
        )

    def days_1_percent_to_impact(self) -> pa.FloatArray:
        return pc.subtract(
            self.orbit.impact_time.mjd(), self.ip_threshold_1_percent.mjd()
        )

    def days_10_percent_to_impact(self) -> pa.FloatArray:
        return pc.subtract(
            self.orbit.impact_time.mjd(), self.ip_threshold_10_percent.mjd()
        )

    def days_50_percent_to_impact(self) -> pa.FloatArray:
        return pc.subtract(
            self.orbit.impact_time.mjd(), self.ip_threshold_50_percent.mjd()
        )

    def days_90_percent_to_impact(self) -> pa.FloatArray:
        return pc.subtract(
            self.orbit.impact_time.mjd(), self.ip_threshold_90_percent.mjd()
        )

    def days_100_percent_to_impact(self) -> pa.FloatArray:
        return pc.subtract(
            self.orbit.impact_time.mjd(), self.ip_threshold_100_percent.mjd()
        )

    def get_diameter_impact_period_data(
        self,
        period_breakdown: Literal["decade", "5year", "year"] = "decade",
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], list]:
        """
        Extract impact time periods and common data needed for diameter-time analysis.

        Parameters
        ----------
        period_breakdown : Literal['decade', '5year', 'year']
            How to break down the impact times:
            - 'decade': Group by 10-year periods (e.g., 2020-2029)
            - '5year': Group by 5-year periods (e.g., 2020-2024)
            - 'year': Group by individual years

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, list]
            A tuple containing:
            - impact_periods: Array of impact periods for each orbit
            - unique_periods: Sorted array of unique periods
            - unique_diameters: Sorted list of unique diameters
        """
        # Extract impact dates and convert to periods based on breakdown
        impact_years = np.array(
            [
                impact_time.datetime.year
                for impact_time in self.orbit.impact_time.to_astropy()
            ]
        )

        if period_breakdown == "decade":
            impact_periods = (
                impact_years // 10
            ) * 10  # Convert year to decade (2023 -> 2020)
        elif period_breakdown == "5year":
            impact_periods = (
                impact_years // 5
            ) * 5  # Convert year to 5-year period (2023 -> 2020)
        else:  # year
            impact_periods = impact_years  # Keep individual years

        unique_periods = np.sort(np.unique(impact_periods))
        unique_diameters = self.orbit.diameter.unique().sort().to_pylist()

        return impact_periods, unique_periods, unique_diameters

    def synodic_period_wrt_earth(self) -> pa.FloatArray:
        """
        Calculate the synodic period in days with respect to Earth for each orbit.

        Assumes the orbit's coordinates are heliocentric. Uses Kepler's Third Law
        to estimate the orbital period from the semi-major axis.

        Returns
        -------
        pa.FloatArray
            The synodic period in days for each orbit. Returns NaN if the
            orbital period is exactly Earth's period.
        """
        # Get Keplerian elements, specifically the semi-major axis (a) in AU
        keplerian_coords = self.orbit.coordinates.to_keplerian()
        a_au = keplerian_coords.a

        # Calculate the orbital period in days using Kepler's Third Law
        # P_days = (2 * pi / k) * a_au^(3/2), where k is the Gaussian gravitational constant
        # P_years = a_au^(3/2)
        # P_days = P_years * EARTH_ORBITAL_PERIOD_DAYS
        object_period_days = pc.multiply(
            pc.power(a_au, 1.5), pa.scalar(EARTH_ORBITAL_PERIOD_DAYS, type=pa.float64())
        )

        # Calculate the difference in mean motion (1/P)
        # Handle cases where the object period might be very close to Earth's period
        delta_mean_motion = pc.subtract(
            pc.divide(1.0, object_period_days),
            pc.divide(1.0, pa.scalar(EARTH_ORBITAL_PERIOD_DAYS, type=pa.float64())),
        )

        # Synodic Period S = 1 / |delta_mean_motion|
        # Use safe_divide to handle potential division by zero (object_period == earth_period)
        # which results in infinity, represented as NaN in floating point.
        synodic_period = pc.divide(1.0, pc.abs(delta_mean_motion))

        # Replace potential infinities (from zero division) with NaN
        is_inf = pc.or_(
            pc.equal(synodic_period, np.inf), pc.equal(synodic_period, -np.inf)
        )
        synodic_period = pc.if_else(is_inf, np.nan, synodic_period)

        return synodic_period


class DiscoveryDates(qv.Table):
    orbit_id = qv.LargeStringColumn()
    discovery_date = Timestamp.as_column(nullable=True)


class WarningTimes(qv.Table):
    orbit_id = qv.LargeStringColumn()
    warning_time = qv.Float64Column(nullable=True)


class DiscoverySummary(qv.Table):
    diameter = qv.Float64Column()
    ast_class = qv.StringColumn()
    discovered = qv.Int64Column()
    total = qv.Int64Column()


class VariantOrbitsWithWindowName(qv.Table):
    window = qv.StringColumn()
    variant = VariantOrbits.as_column()


class OrbitWithWindowName(qv.Table):
    window = qv.StringColumn()
    orbit = Orbits.as_column()


@dataclass
class RunConfiguration:
    # How many variants to create for each orbit
    monte_carlo_samples: int
    assist_epsilon: float
    assist_min_dt: float
    assist_initial_dt: float
    assist_adaptive_mode: int
    seed: int
    pointing_database_file: str
    max_processes: Optional[int] = None

    @classmethod
    def from_json(cls, json_file: str) -> "RunConfiguration":
        with open(json_file, "r") as f:
            return cls(**json.load(f))

    def to_json(self, json_file: str) -> None:
        with open(json_file, "w") as f:
            json.dump(asdict(self), f)
