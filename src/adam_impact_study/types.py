import json
from dataclasses import asdict, dataclass
from typing import Optional

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
    diameter = qv.Float64Column()
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
    # Time when observations met minimum artificial discovery criteria
    # Currently set to 3 unique nights of tracklets
    discovery_time = Timestamp.as_column(nullable=True)
    # Duration in days since the first non-zero impact probability
    # until true impact date
    warning_time = qv.Float64Column(nullable=True)
    # Time between discovery and non-zero impact probability
    # (note, this is partially a function of our monte-carlo sampling)
    realization_time = qv.Float64Column(nullable=True)
    # How close all the windows got to discovering the definite impact nature
    maximum_impact_probability = qv.Float64Column(nullable=True)
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
