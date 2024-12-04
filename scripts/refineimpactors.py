from adam_core.orbits import Orbits
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
import os

import numpy as np

def sample_random_rows(orbits, sample_size=1000):
    random_indices = np.random.choice(len(orbits), size=sample_size, replace=False)
    mask = np.zeros(len(orbits), dtype=bool)
    mask[random_indices] = True
    sampled_orbits = orbits.apply_mask(mask)
    
    return sampled_orbits

inputdir = './Impactors_Outputs/'

for file in os.listdir(inputdir):
    if "impacting_objects" not in file:
        continue
    print(file)

    orbits = Orbits.from_parquet(f"{inputdir}/{file}")

    print(len(orbits))
    assert len(orbits) > 1000

    sampled_orbits = sample_random_rows(orbits, 1000)

    print(len(sampled_orbits))
    assert len(sampled_orbits) == 1000

    sampled_orbits.to_parquet(f"final_sampled/sampled_{file}")