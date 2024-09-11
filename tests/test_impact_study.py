import pandas as pd
import pyarrow.compute as pc
from adam_core.orbits import Orbits

from adam_core.orbits.query import query_horizons
from adam_core.time import Timestamp
from adam_core.propagator.adam_assist import ASSISTPropagator
from adam_core.dynamics.impacts import calculate_impact_probabilities, calculate_impacts
from adam_impact_study.sorcha_utils import run_sorcha
from adam_impact_study.fo_od import run_fo_od
from adam_impact_study.conversions import impactor_file_to_adam_orbit

#test all impactors hit on specific date




def wip_test_for_impacts():
    propagator = ASSISTPropagator()
    impactors_file = "../demo/data/10_impactors.csv"
    initial_orbit_objects = impactor_file_to_adam_orbit(impactors_file)
    for obj_id in initial_orbit_objects:
        print(obj_id)
        impactor_orbit_object = obj_id
        print("Expected impact date: ", impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0])
        results, impacts = propagator.detect_impacts(impactor_orbit_object, 60)
        print("Impact date: ", impacts.coordinates.time.to_numpy()[0])
        assert abs(impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0] - impacts.coordinates.time.to_numpy()[0]) < 1
        #assert impactor_orbit_object.coordinates.time.add_days(30).to_numpy()[0] == impacts.coordinates.time.to_numpy()[0]

#test sorcha obs with 


def wip_test_state_vector_vs_horizon():

    objects = ["Apophis"]
    times = Timestamp.from_mjd([59000], scale="tdb")

    horizon_orbits = query_horizons(objects, times)

    propagator = ASSISTPropagator()

    sorcha_physical_params_string = "15.88 1.72 0.48 -0.11 -0.12 -0.12 0.15"
    physical_params_list = [
        float(param) for param in sorcha_physical_params_string.split()
    ]
    data = []
    for obj_id in horizon_orbits.object_id:
        data.append(
            {
                "ObjID": str(obj_id),
                "H_r": physical_params_list[0],
                "u-r": physical_params_list[1],
                "g-r": physical_params_list[2],
                "i-r": physical_params_list[3],
                "z-r": physical_params_list[4],
                "y-r": physical_params_list[5],
                "GS": physical_params_list[6],
            }
        )
    physical_params_df = pd.DataFrame(data)

    # Run Sorcha to generate observational data
    od_observations = run_sorcha(
        horizon_orbits,
        sorcha_config_file,
        sorcha_orbits_file,
        sorcha_physical_params_file,
        sorcha_output_file,
        physical_params_df,
        pointing_file,
        sorcha_output_name,
        "results",
    )
    if od_observations is None:
        return None

    # Iterate over each object and calculate impact probabilities
    object_ids = od_observations.object_id.unique()
    impact_results = None
    for obj in object_ids:
        print("Object ID: ", obj)
        od_obs = od_observations.apply_mask(pc.equal(od_observations.object_id, obj))
        fo_file_name = f"fo_output_{obj}.csv"
        fo_output_folder = f"fo_output_{obj}"
        od_observations_to_ades_file(filtered_obs, f"{RESULT_DIR}/{fo_file_name}")

        fo_orbit = run_fo_od(
            fo_file_name,
            fo_output_folder,
            FO_DIR,
            ".",
            "results",
        )

        propagated_orbit = propagator.propagate_orbits(
            fo_orbit, horizon_orbits.coordinates.time, covariance=True, num_samples=1000
        )

    return impact_results

        


#wip_test_state_vector_vs_horizon()
wip_test_for_impacts()