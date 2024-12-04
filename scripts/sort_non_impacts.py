from adam_core.propagator.adam_assist import (
    ASSISTPropagator,
#    download_jpl_ephemeris_files,
)
from adam_core.time import Timestamp
from adam_core.orbits import Orbits
import quivr as qv

from adam_impact_study.conversions import impactor_file_to_adam_orbit
import pyarrow.compute as pc
import pyarrow as pa
import os

inputdir = './results_raw/'
outputdir = './processed_results/'
files = []

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

for file in os.listdir(inputdir):

    print(file)

    files.append(f"{inputdir}/{file}")

    no_impact = []
    impact_orb = []
    no_impact_initial = []
    back_propagated_orbits = Orbits.empty()
    round_trip_orbits = Orbits.empty()

#    download_jpl_ephemeris_files()
    propagator = ASSISTPropagator()
    initial_orbit_objects = impactor_file_to_adam_orbit(f"{inputdir}/{file}")

    print(initial_orbit_objects)
    print(len(initial_orbit_objects))

    initial_orbit_objects.to_parquet(f"{file.split('.csv')[0]}_initial_objects.parquet")

    for obj_id in initial_orbit_objects:
        print(obj_id.orbit_id)
        print("Initial Orbit Object:", obj_id.coordinates.values)
        print("Initial Orbit Object Time:", obj_id.coordinates.time.to_numpy()[0])
        impactor_orbit_object = obj_id
        results, impacts = propagator.detect_impacts(impactor_orbit_object, 60)
        if len(impacts) == 0:
            no_impact_initial.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
            no_impact.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
            print(f"No 30 day impact for object {obj_id.orbit_id}")
            continue
        else:
            print(impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0])
            print(impacts.coordinates.time.to_numpy()[0])
            print(
                abs(
                    impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                    - impacts.coordinates.time.to_numpy()[0]
                )
            )
            if abs(
                impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                - impacts.coordinates.time.to_numpy()[0]
            ) > 1:
                print("#### Unusually large time difference ####")
        time_start = Timestamp.from_jd([2460800.5])
        results_back = propagator.propagate_orbits(impactor_orbit_object, time_start, covariance=True)
        print("Orbit at start:", results_back[0].coordinates.values)
        print("Orbit at start Time:", results_back[0].coordinates.time.to_numpy()[0])
        if back_propagated_orbits is None:
            back_propagated_orbits = results_back
        else:
            back_propagated_orbits = qv.concatenate([back_propagated_orbits, results_back])
        results_forward = propagator.propagate_orbits(results_back, impactor_orbit_object.coordinates.time, covariance=True)
        print("Orbit forward:", results_forward[0].coordinates.values)
        print("Orbit forward Time:", results_forward[0].coordinates.time.to_numpy()[0])
        if round_trip_orbits is None:
            round_trip_orbits = results_forward
        else:
            round_trip_orbits = qv.concatenate([round_trip_orbits, results_forward])
        results, impacts = propagator.detect_impacts(results_forward[0], 60)
        if len(impacts) == 0:
            print("No impacts for object ", obj_id.orbit_id)
            no_impact.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
        else:
            impact_orb.append(obj_id.orbit_id.to_numpy(zero_copy_only=False)[0])
            print(impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0])
            print(impacts.coordinates.time.to_numpy()[0])
            print(
                abs(
                    impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                    - impacts.coordinates.time.to_numpy()[0]
                )
            )
            if abs(
                impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0]
                - impacts.coordinates.time.to_numpy()[0]
            ) > 1:
                print("#### Unusually large time difference ####")

    print("No impact for objects:", no_impact)
    print("Impacts for objects:", impact_orb)
    print("No impact for initial objects:", no_impact_initial)

    print("Number of objects with no impact:", len(no_impact))
    print("Number of objects with impact:", len(impact_orb))
    print("Number of objects with no impact for initial objects:", len(no_impact_initial))

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