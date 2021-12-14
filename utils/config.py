import numpy as np
from utils import CROP_ENCODING

RANDOM_SEED = 16
IMG_SIZE = 366

AUTHORS = 'Papoutsis I., Sykas D., Zografakis D., Sdraka M.'
DATASET_VERSION = '21.03'
DATASET_URL = 'https://www.sen4agrinet.space.noa.gr/'

LICENSES = [
    {
        'url': 'https://creativecommons.org/licenses/by-sa/4.0/',
        'id': 1,
        'name': 'Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)'
    },
]


# Band names and their resolutions
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

# Maps arbitrary number of classes to a discrete range of numbers starting from 0
LINEAR_ENCODER = {val: i + 1 for i, val in enumerate(sorted(list(CROP_ENCODING.values())))}
LINEAR_ENCODER[0] = 0

# Extract patches based on this band
REFERENCE_BAND = 'B02'

# Output DTYPE for patches (bands), range is 0-10k, use smaller int to reduce size required
BAND_OUT_DTYPE = np.uint16

# Output DTYPE for patches (labels/parcels), use bigger ints cause parcels have greater range
LABEL_OUT_DTYPE = np.uint32

# Divider for normalizing tiff data to [0-1] range
NORMALIZATION_DIV = 10000
