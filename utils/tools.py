import copy
import numpy as np
from typing import Union
from utils import NORMALIZATION_DIV


def hollstein_mask(
        bands: Union[dict, np.ndarray],
        clouds: bool = True,
        cirrus: bool = False,
        shadows: bool = False,
        snow: bool = False,
        requires_norm: bool = False,
        reference_bands=None
) -> np.ndarray:

    # BASED: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/hollstein
    bands = copy.deepcopy(bands)

    # If ndarray is given, assume NxBxWxH (Time bins, Bands, Width, Height) and convert it to dictionary
    # else assume it is a dictionary
    if not isinstance(bands, dict):
        assert reference_bands is not None, 'Hollstein: ndarray was given, but reference bands were empty'
        bands = {band: bands[:, i, :, :] for i, band in enumerate(reference_bands)}

    # This should be the reverse of requires_norm in dataloader
    if not requires_norm:
        bands = {key:  band / NORMALIZATION_DIV for key, band in bands.items()}

    out = {}

    if shadows:
        shadows_cond = ((bands['B03'] < 0.319) & (bands['B8A'] < 0.166) & (
                ((bands['B03'] - bands['B07'] < 0.027) & (bands['B09'] - bands['B11'] >= -0.097)) |
                ((bands['B03'] - bands['B07'] >= 0.027) & (bands['B09'] - bands['B11'] >= 0.021))
        )) | \
                       ((bands['B03'] >= 0.319) & (np.divide(bands['B05'], bands['B11']) >= 4.33) &
                        (bands['B03'] < 0.525) & (np.divide(bands['B01'], bands['B05']) >= 1.184))

        out['shadow_mask'] = shadows_cond

    if clouds:
        clouds_cond = (
                (bands['B03'] >= 0.319) & (bands['B05'] / bands['B11'] < 4.33) &
                (
                        ((bands['B11'] - bands['B10'] < 0.255) & (bands['B06'] - bands['B07'] < -0.016)) |
                        ((bands['B11'] - bands['B10'] >= 0.255) & (bands['B01'] >= 0.3))
                )
        )

        out['cloud_mask'] = clouds_cond

    if cirrus:
        cirrus_cond = (
            (
                (bands['B03'] < 0.319) & (bands['B8A'] >= 0.166) & (np.divide(bands['B02'], bands['B10']) < 14.689) &
                (np.divide(bands['B02'], bands['B09']) >= 0.788)
            ) |
            (
                (bands['B03'] >= 0.319) & (np.divide(bands['B05'], bands['B11']) < 4.33) &
                (bands['B11'] - bands['B10'] < 0.255) & (bands['B06'] - bands['B07'] >= -0.016)
            )
        )

        out['cirrus_mask'] = cirrus_cond

    if snow:
        snow_cond = (
                (bands['B03'] >= 0.319) & (np.divide(bands['B05'], bands['B11']) >= 4.33) & (bands['B03'] >= 0.525)
        )

        out['snow_mask'] = snow_cond

    # Logical OR between all masks is the final mask
    out = np.any(np.array(list(out.values())), axis=0)

    return out
