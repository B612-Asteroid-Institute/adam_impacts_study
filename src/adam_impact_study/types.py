import quivr as qv
from adam_core.coordinates import SphericalCoordinates
from adam_core.observers import Observers
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


class ImpactStudyResults(qv.Table):
    object_id = qv.LargeStringColumn()
    observation_start = Timestamp.as_column()
    observation_end = Timestamp.as_column()
    observation_count = qv.UInt64Column()
    observations_rejected = qv.UInt64Column()
    observation_nights = qv.UInt64Column()
    impact_probability = qv.Float64Column(nullable=True)
    error = qv.LargeStringColumn(nullable=True)
