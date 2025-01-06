import os

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.orbits import Orbits
from adam_core.propagator.adam_assist import (  # download_jpl_ephemeris_files,
    ASSISTPropagator,
)
from adam_core.time import Timestamp

from adam_impact_study.conversions import impactor_file_to_adam_orbit

inputdir = './results_raw/'
outputdir = './processed_results/'
files = []

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

for file in os.listdir(inputdir):

    logger.info(file)

    files.append(f"{inputdir}/{file}")

    no_impact = []
    impact_orb = []
    no_impact_initial = []
    back_propagated_orbits = Orbits.empty()
    round_trip_orbits = Orbits.empty()

#    download_jpl_ephemeris_files()
    propagator = ASSISTPropagator()
    initial_orbit_objects = impactor_file_to_adam_orbit(f"{inputdir}/{file}")

    logger.info(initial_orbit_objects)
    logger.info(len(initial_orbit_objects))

    initial_orbit_objects.to_parquet(f"{file.split('.csv')[0]}_initial_objects.parquet")

    for obj_id in initial_orbit_objects:
        logger.info(obj_id.orbit_id)
        logger.info("Initial Orbit Object:", obj_id.coordinates.values)
        logger.info("Initial Orbit Object Time:", obj_id.coordinates.time.to_numpy()[0])
        impactor_orbit_object = obj_id
        results, impacts = propagator.detect_impacts(impactor_orbit_object, 60)
        if len(impacts) == 0:
            no_impact_initial.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
            no_impact.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
            logger.info(f"No 30 day impact for object {obj_id.orbit_id}")
            continue
        else:
            logger.info(impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0])
            logger.info(impacts.coordinates.time.to_numpy()[0])
            logger.info(
                abs(
                    impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                    - impacts.coordinates.time.to_numpy()[0]
                )
            )
            if abs(
                impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                - impacts.coordinates.time.to_numpy()[0]
            ) > 1:
                logger.info("#### Unusually large time difference ####")
        time_start = Timestamp.from_jd([2460800.5])
        results_back = propagator.propagate_orbits(impactor_orbit_object, time_start, covariance=True)
        logger.info("Orbit at start:", results_back[0].coordinates.values)
        logger.info("Orbit at start Time:", results_back[0].coordinates.time.to_numpy()[0])
        if back_propagated_orbits is None:
            back_propagated_orbits = results_back
        else:
            back_propagated_orbits = qv.concatenate([back_propagated_orbits, results_back])
        results_forward = propagator.propagate_orbits(results_back, impactor_orbit_object.coordinates.time, covariance=True)
        logger.info("Orbit forward:", results_forward[0].coordinates.values)
        logger.info("Orbit forward Time:", results_forward[0].coordinates.time.to_numpy()[0])
        if round_trip_orbits is None:
            round_trip_orbits = results_forward
        else:
            round_trip_orbits = qv.concatenate([round_trip_orbits, results_forward])
        results, impacts = propagator.detect_impacts(results_forward[0], 60)
        if len(impacts) == 0:
            logger.info("No impacts for object ", obj_id.orbit_id)
            no_impact.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
        else:
            impact_orb.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
            logger.info(impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0])
            logger.info(impacts.coordinates.time.to_numpy()[0])
            logger.info(
                abs(
                    impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                    - impacts.coordinates.time.to_numpy()[0]
                )
            )
            if abs(
                impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                - impacts.coordinates.time.to_numpy()[0]
            ) > 1:
                logger.info("#### Unusually large time difference ####")

    logger.info("No impact for objects:", no_impact)
    logger.info("Impacts for objects:", impact_orb)
    logger.info("No impact for initial objects:", no_impact_initial)

    logger.info("Number of objects with no impact:", len(no_impact))
    logger.info("Number of objects with impact:", len(impact_orb))
    logger.info("Number of objects with no impact for initial objects:", len(no_impact_initial))

    impact_orb_pa = pa.array(impact_orb)
    impacting_mask = pc.is_in(initial_orbit_objects.orbit_id, impact_orb_pa)
    impacting_objects = initial_orbit_objects.apply_mask(impacting_mask)
    no_impact_pa = pa.array(no_impact)
    non_impacting_mask = pc.is_in(initial_orbit_objects.orbit_id, no_impact_pa)
    non_impacting_objects = initial_orbit_objects.apply_mask(non_impacting_mask)
    impacting_objects.to_parquet(f"{outputdir}/{file.split('.csv')[0]}_impacting_objects.parquet")
    non_impacting_objects.to_parquet(f"{outputdir}/{file.split('.csv')[0]}_non_impacting.parquet")
    back_propagated_orbits.to_parquet(f"{outputdir}/{file.split('.csv')[0]}_back_propagated_orbits.parquet")
    round_trip_orbits.to_parquet(f"{outputdir}/{file.split('.csv')[0]}_round_trip_orbits.parquet")

print (files)