import quivr as qv
from adam_core.coordinates import (
    CartesianCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
)
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp


class Photometry(qv.Table):
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn()


class Observations(qv.Table):
    obs_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn()
    coordinates = SphericalCoordinates.as_column()
    observers = Observers.as_column()
    photometry = Photometry.as_column(nullable=True)
    observing_night = qv.Int64Column(nullable=True)


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

    def photometric_properties(self) -> "PhotometricProperties":
        from .sorcha_utils import PhotometricProperties

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
    observation_start = Timestamp.as_column()
    observation_end = Timestamp.as_column()
    observation_count = qv.UInt64Column()
    observations_rejected = qv.UInt64Column()
    observation_nights = qv.UInt64Column()
    impact_probability = qv.Float64Column(nullable=True)
    impact_time = Timestamp.as_column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)
    car_coordinates = CartesianCoordinates.as_column(nullable=True)
    kep_coordinates = KeplerianCoordinates.as_column(nullable=True)


class ImpactorResultSummary(qv.Table):
    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    impact_time = Timestamp.as_column()
    # Number of distinct orbit fitting windows
    windows = qv.Int64Column()
    # Number of distinct nights of observations
    nights = qv.Int64Column()
    # Number of distinct observations recovered
    observations = qv.Int64Column()
    # Number of singletons from observations recovered
    singletons = qv.Int64Column()
    # Number of tracklets from observations recovered
    tracklets = qv.Int64Column()
    # Whether the object was observed at all
    observed = qv.BooleanColumn(default=False)
    # Whether the object was discovered during the run
    discovered = qv.BooleanColumn(default=False)
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
