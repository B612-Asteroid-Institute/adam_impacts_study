from typing import Optional


def get_study_paths(base_dir: str, run_name: str, object_id: str, time_range: Optional[str] = None) -> dict:
    """Get standardized paths for impact study results.
    
    Parameters
    ----------
    base_dir : str
        Base directory for all results
    run_name : str
        Name of the study run
    object_id : str
        Object identifier
    time_range : str, optional
        Time range in format "mjd_start__mjd_end"
        
    Returns
    -------
    dict
        Dictionary containing all relevant paths
    """
    import os
    
    run_dir = os.path.join(base_dir, run_name)
    obj_dir = os.path.join(run_dir, object_id)
    
    paths = {
        'base': obj_dir,
        'sorcha_inputs': os.path.join(obj_dir, 'sorcha_inputs'),
        'sorcha_outputs': os.path.join(obj_dir, 'sorcha_outputs'),
    }
    
    if time_range:
        time_dir = os.path.join(obj_dir, time_range)
        paths.update({
            'fo_inputs': os.path.join(time_dir, 'fo_inputs'),
            'fo_outputs': os.path.join(time_dir, 'fo_outputs'),
            'propagated': os.path.join(time_dir, 'propagated'),
            'results': os.path.join(time_dir, 'results.parquet')
        })
    
    # Create directories
    for path in paths.values():
        if not path.endswith('.parquet'):
            os.makedirs(path, exist_ok=True)
            
    return paths 