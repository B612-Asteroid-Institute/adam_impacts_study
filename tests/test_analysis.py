import os

import pytest
from adam_core.time import Timestamp

from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.impacts_study import ImpactStudyResults


@pytest.fixture
def impact_study_results():
    object_ids = ["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"]
    start_dates = Timestamp.from_mjd([59800.0, 59800.0, 59800.0, 59800.0, 59800.0, 59800.0])
    end_dates = Timestamp.from_mjd([59801.0, 59802.0, 59803.0, 59801.0, 59802.0, 59803.0])
    observation_counts = [10, 20, 30, 10, 20, 30]
    observation_nights = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    impact_probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    impact_result = ImpactStudyResults.from_kwargs(
        object_id=object_ids,
        observation_start=start_dates,
        observation_end=end_dates,
        observation_count=observation_counts,
        observation_nights=observation_nights,
        impact_probability=impact_probabilities,
    )

    return impact_result


def test_plot_ip_over_time(impact_study_results, tmpdir):
    tmpdir_path = tmpdir.mkdir("plots")
    os.chdir(tmpdir_path)
    plot_ip_over_time(impact_study_results)
    object_ids = impact_study_results.object_id.unique()
    for obj_id in object_ids:
        assert os.path.exists(f"IP_{obj_id}.png")
