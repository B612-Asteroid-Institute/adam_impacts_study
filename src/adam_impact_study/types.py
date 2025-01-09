import quivr as qv
from adam_core.coordinates import CartesianCoordinates, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from .physical_params import PhotometricProperties


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
    dynamical_class = qv.LargeStringColumn()  # AMOR / ATEN / APO
    photometric_properties = PhotometricProperties.as_column()
    ast_class = qv.LargeStringColumn(nullable=True)  # not nullable in production
    albedo = qv.Float64Column(nullable=True)  # not nullable in production

    def to_orbits(self) -> Orbits:
        return Orbits.from_kwargs(
            orbit_id=self.orbit_id,
            object_id=self.object_id,
            coordinates=self.coordinates,
        )


class ImpactStudyResults(qv.Table):
    object_id = qv.LargeStringColumn()
    observation_start = Timestamp.as_column()
    observation_end = Timestamp.as_column()
    observation_count = qv.UInt64Column()
    observations_rejected = qv.UInt64Column()
    observation_nights = qv.UInt64Column()
    impact_probability = qv.Float64Column(nullable=True)
    impact_time = Timestamp.as_column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)


class ImpactorResults(qv.Table):
    object_id = qv.LargeStringColumn()
    impact_time = Timestamp.as_column()
    windows = qv.Int64Column()
    nights = qv.Int64Column()
    observations = qv.Int64Column()
    singletons = qv.Int64Column()
    tracklets = qv.Int64Column()
    observed = qv.BooleanColumn(default=False)
    discovered = qv.BooleanColumn(default=False)
    discovery_time = Timestamp.as_column(nullable=True)
    warning_time = qv.Float64Column(nullable=True)
    realization_time = qv.Float64Column(nullable=True)
    maximum_impact_probability = qv.Float64Column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)
