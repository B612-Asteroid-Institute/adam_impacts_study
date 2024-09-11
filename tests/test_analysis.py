import os
from adam_impact_study.analysis import plot_ip_over_time
from adam_impact_study.impacts_study import ImpactStudyResults
import pytest

@pytest.fixture
def impact_study_results():
    # Create sample data using quiver's Table class
    object_ids = ["obj1", "obj1", "obj1", "obj2", "obj2", "obj2"]
    days = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    impact_probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    impact_result = ImpactStudyResults.from_kwargs(
        object_id=object_ids,
        day=days,
        impact_probability=impact_probabilities,
    )

    return impact_result

def test_plot_ip_over_time(impact_study_results, tmpdir):
    tmpdir_path = tmpdir.mkdir("plots")
    os.chdir(tmpdir_path)
    plot_ip_over_time(impact_study_results)
    object_ids = impact_study_results.object_id.unique()
    for obj_id in object_ids:
        assert os.path.exists(f"IP_{obj_id}.png"), f"Plot for object {obj_id} was not created"
