import logging
import os
from typing import Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from adam_core.observations.ades import (
    ADES_to_string,
    ADESObservations,
    ObsContext,
    ObservatoryObsContext,
    SubmitterObsContext,
    TelescopeObsContext,
)
from adam_core.orbits import Orbits
from adam_fo import fo

from .types import Observations

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("ADAM_LOG_LEVEL", "INFO"))


fo_file_whitelist = [
    "total.json",
    "covar.json",
    "elem_short.json",
    "debug.txt" "observations.ades",
    "environ.dat",
]


def observations_to_ades(observations: Observations) -> Tuple[str, ADESObservations]:
    """
    Convert Observations to ADES format string.

    Parameters
    ----------
    observations : Observations
        Observations to convert

    Returns
    -------
    Tuple[str, ADESObservations]
        Tuple containing:
        - ADES format string
        - ADESObservations object
    """
    # Extract the original orbit_id since trkSub has an 8-character limit
    orbit_ids = observations.orbit_id

    # Convert uncertainties in RA and Dec to arcseconds
    # and adjust the RA uncertainty to account for the cosine of the declination
    sigma_ra_cos_dec = (
        np.cos(np.radians(observations.coordinates.lat.to_numpy(zero_copy_only=False)))
        * observations.coordinates.covariance.sigmas[:, 1]
    )
    sigma_ra_cos_dec_arcseconds = pa.array(sigma_ra_cos_dec * 3600, type=pa.float64())
    sigma_dec_arcseconds = pa.array(
        observations.coordinates.covariance.sigmas[:, 2] * 3600, type=pa.float64()
    )

    # Replace nans with nulls using pyarrow
    sigma_ra_cos_dec_arcseconds = pc.if_else(
        pc.is_nan(sigma_ra_cos_dec_arcseconds), None, sigma_ra_cos_dec_arcseconds
    )
    sigma_dec_arcseconds = pc.if_else(
        pc.is_nan(sigma_dec_arcseconds), None, sigma_dec_arcseconds
    )

    # Serialize observations to an ADES table
    ades_observations = ADESObservations.from_kwargs(
        trkSub=pc.utf8_slice_codeunits(orbit_ids, 0, 8),
        obsTime=observations.coordinates.time,
        ra=observations.coordinates.lon,
        dec=observations.coordinates.lat,
        rmsRACosDec=sigma_ra_cos_dec_arcseconds,
        rmsDec=sigma_dec_arcseconds,
        mag=observations.photometry.mag,
        rmsMag=observations.photometry.mag_sigma,
        band=observations.photometry.filter,
        stn=pa.repeat("X05", len(observations)),
        mode=pa.repeat("NA", len(observations)),
        astCat=pa.repeat("NA", len(observations)),
    )

    # Minimal obscontext representing our simulated X05 survey
    obs_contexts = {
        "X05": ObsContext(
            observatory=ObservatoryObsContext(
                mpcCode="X05", name="Vera C. Rubin Observatory - LSST"
            ),
            submitter=SubmitterObsContext(
                name="K. Kiker",
                institution="B612 Asteroid Institute",
            ),
            observers=["J. Doe"],
            measurers=["J. Doe"],
            telescope=TelescopeObsContext(
                name="Simonyi Survey Telescope",
                design="Reflector",
            ),
        ),
    }

    ades_string = ADES_to_string(ades_observations, obs_contexts)
    return ades_string, ades_observations


def run_fo_od(
    observations: Observations,
    fo_result_dir: str,
    fo_dir_cleanup: bool = True,
) -> Tuple[Orbits, ADESObservations, Optional[str]]:
    """Run Find_Orb orbit determination with directory-based paths

    Parameters
    ----------
    observations : Observations
        Observations to process
    fo_result_dir : str
        Directory where Find_Orb output files will be written

    Returns
    -------
    Tuple[Orbits, ADESObservations, Optional[str]]
        Tuple containing:
        - Determined orbit
        - Processed observations
        - Error message (if any)
    """
    # This function is only valid for a single orbit_id
    if len(observations.orbit_id.unique()) > 1:
        raise ValueError("This function is only valid for a single orbit_id")

    ades_string, rejected = observations_to_ades(observations)

    min_mjd = observations.coordinates.time.min().mjd()[0].as_py()
    max_mjd = observations.coordinates.time.max().mjd()[0].as_py()
    logger.info(
        f"Running fo for {observations.orbit_id[0].as_py()} from {min_mjd} to {max_mjd}"
    )

    orbit, rejected, error = fo(
        ades_string,
        out_dir=fo_result_dir,
        # Ignore clean_up in favor of keeping whitelisted files below
        clean_up=False,
    )

    if fo_dir_cleanup:
        for file in os.listdir(fo_result_dir):
            if file not in fo_file_whitelist:
                os.remove(os.path.join(fo_result_dir, file))

    # Re-assign orbit_id to the original value if we found an orbit
    if len(orbit) > 0:
        orbit = orbit.set_column("orbit_id", observations.orbit_id[:1])

    return orbit, rejected, error
