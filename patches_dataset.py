import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Tuple, Union
from tqdm import tqdm

import xarray as xr
from pycocotools.coco import COCO
import netCDF4

from torch.utils.data import Dataset
from pathlib import Path
import pytorch_lightning as pl

from utils.config import RANDOM_SEED, BANDS, IMG_SIZE, REFERENCE_BAND, NORMALIZATION_DIV
from utils.tools import hollstein_mask

# Set seed for everything
pl.seed_everything(RANDOM_SEED)


class PatchesDataset(Dataset):
    '''
    PyTorch Dataset for iterating over batches of data.

    It constructs a data batch by grouping together images in time bins based on
    a given frequency and then calculating the median of each bin,
    e.g. if given frequency is "1 month" then the median of all images of the same month
    is calculated and used as a sample.

    If data for a time bin is absent then the median of all data from previous and
    following bins is calculated.
    E.g. if time frequency is "1 month" and there are no images for March, then
    the median of all images from February and April is calculated and used.
    In the case where there are no data before or after that specific bin, then
    the following or previous median is replicated respectively.
    E.g. if time frequency is "1 month" and there are no images for January, then
    the median of February is used for this month too.

    Implements lazy reading, to avoid high memory usage. First it creates an index
    file containing all paths for the available patches and only reads the image
    files when batching.
    '''

    def __init__(
            self,
            coco: COCO,
            root_path_netcdf: Union[str, Path] = None,
            bands: list = None,
            group_freq: str = '1MS',
            saved_medians: bool = False,
            linear_encoder: dict = None,
            prefix: str = None,
            window_len: int = 12,
            fixed_window: bool = False,
            requires_norm: bool = True,
            return_masks: bool = False,
            clouds: bool = True,
            cirrus: bool = True,
            shadow: bool = True,
            snow: bool = True,
            output_size: tuple = None,
            binary_labels: bool = False,
            mode: str = None,
            return_parcels: bool = False
    ) -> None:
        '''
        Parameters
        ----------
        coco: COCO Object
            A COCO object containing the data.
        root_path_netcdf: Path or str, default None
            The path containing the netcdf files.
        bands: list of str, default None
            A list of the bands to use. If None, then all available bands are
            taken into consideration. Note that the bands are given in a two-digit
            format, e.g. '01', '02', '8A', etc.
        group_freq: str, default '1MS'
            The frequency to use for binning. All Pandas offset aliases are supported.
            Check: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        saved_medians: boolean, default False
            Whether to precompute and save all medians. This saves on computation
            time during batching.
        linear_encoder: dict, default None
            Maps arbitrary crop_ids to range 0-len(unique(crop_id)).
        prefix: str, default None
            A prefix to use for all exported files. If None, then the current
            timestamp is used.
        window_len: integer, default 12
            If a value is passed, then a rolling window of this length is applied
            over the data. E.g. if `window_len` = 6 and `group_freq` = '1M', then
            a 6-month rolling window will be applied and each batch will contain
            6 months of training data and the corresponding label.
        fixed_window: boolean, default False
            If True, then a fixed window including months 4 (April) to 9 (September)
            is used instead of a rolling one.
        requires_norm: boolean, default True
            If True, then it normalizes the dataset to [0, 1] range.
        return_masks: boolean, default False
            based: https://github.com/sentinel-hub/custom-scripts/tree/master/sentinel-2/hollstein
            If True, returns Logical OR of all hollstein masks, see below.
        clouds: boolean, default True
            If True and return_masks=True, returns mask for clouds
        cirrus: boolean, default True
            If True and return_masks=True, returns mask for cirrus
        shadow: boolean, default True
            If True and return_masks=True, returns mask for shadow
        snow: boolean, default True
            If True and return_masks=True, returns mask for snow
        output_size: tuple of int, default None
            If a tuple (H, W) is given, then the output images will be divided
            into non-overlapping subpatches of size (H, W). Otherwise, the images
            will retain their original size.
        binary_labels: bool, default False
            Map categories to 0 background, 1 parcel.
        mode: str, ['train', 'val', 'test']
            The running mode. Used to determine the correct path for the median files.
        return_parcels: boolean, default False
            If True, then a boolean mask for the parcels is also returned.
        '''

        if prefix is None:
            prefix = datetime.now().strftime("%Y%m%d%H%M%S")

        # First Keep local copies
        self.coco = coco
        self.patch_ids = list(sorted(self.coco.imgs.keys()))

        if root_path_netcdf is not None:
            self.root_path_netcdf = Path(root_path_netcdf)
        else:
            self.root_path_netcdf = None

        # number of total patches is given by number of patches in coco
        self.num_patches = len(self.patch_ids)

        self.patch_width, self.patch_height = IMG_SIZE, IMG_SIZE
        self.padded_patch_width, self.padded_patch_height = IMG_SIZE, IMG_SIZE

        self.bands = sorted(bands)
        self.num_bands = len(self.bands)
        self.binary_labels = binary_labels

        self.return_masks = return_masks
        self.masks = {}

        if return_masks:
            # Assert all required bands are given in order to calculate hollstein
            required_bands = ['B01', 'B02', 'B03', 'B05', 'B06', 'B07', 'B09', 'B10', 'B11', 'B8A']
            missing_bands = [band for band in required_bands if band not in self.bands]
            assert len(missing_bands) == 0, f'bands: "{missing_bands}" are missing'

            self.masks = {
                'clouds': clouds,
                'cirrus': cirrus,
                'shadow': shadow,
                'snow': snow
            }

        self.img_size = IMG_SIZE
        self.requires_pad = False
        self.requires_subpatching = False

        self.return_parcels = return_parcels

        if output_size is None:
            self.output_size = [self.img_size, self.img_size]

        assert isinstance(output_size[0], int) and isinstance(output_size[1], int),\
            'sub-patches dims must be integers!'

        assert output_size[0] == output_size[1], \
            f'Only square sub-patch size is supported. Mismatch: {output_size[0]} != {output_size[1]}.'

        self.output_size = [int(dim) for dim in output_size]

        # Calculate number of sub-patches in each dimension, check if image needs to be padded
        # TODO: Replace padding with PyTorch Transforms so we can use it to update annotations accordingly
        # Ref: https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py#L115
        if self.output_size[0] < self.patch_height or self.output_size[1] < self.patch_width:
            # If output_size is smaller than patch dims, then we need to break it down to subpatches
            self.requires_subpatching = True

        # Calculating padding offsets if there is a need to
        # In case `saved_medians` is True, then we assume that the medians have already
        # taken padding into account during computation
        if not saved_medians and \
            ((self.patch_height % self.output_size[0] != 0) or (self.patch_width % self.output_size[1] != 0)):
            self.requires_pad = True
            self.pad_top, self.pad_bot, self.pad_left, self.pad_right = self.get_padding_offset()

            # patch_height should always match patch_width because we have square images,
            # but doing it like this ensures expandability
            self.padded_patch_height += (self.pad_top + self.pad_bot)
            self.padded_patch_width += (self.pad_left + self.pad_right)

        self.num_subpatches = (self.padded_patch_height // self.output_size[0]) * (self.padded_patch_width // self.output_size[1])

        self.requires_norm = requires_norm

        self.group_freq = group_freq
        self.window_len = window_len
        self.fixed_window = fixed_window
        self.linear_encoder = linear_encoder

        # Dtypes
        self.medians_dtype = np.float32
        self.label_dtype = np.int64

        # We index based on year, number of bins should be the same for every year
        # therefore, calculate them using a random year
        self.num_buckets = len(pd.date_range(start=f'2020-01-01', end=f'2021-01-01', freq=self.group_freq)) - 1

        self.saved_medians = saved_medians
        self.medians_dir = Path(f'logs/medians/{prefix}_medians_{group_freq}_{"".join(self.bands)}/{mode}')


    def get_padding_offset(self):
        img_size_x = self.patch_height
        img_size_y = self.patch_width

        output_size_x = self.output_size[0]
        output_size_y = self.output_size[1]

        # Calculate padding offset
        if img_size_x >= output_size_x:
            pad_x = int(output_size_x - img_size_x % output_size_x)
        else:
            # For bigger images, is just the difference
            pad_x = output_size_x - img_size_x

        if img_size_y >= output_size_y:
            pad_y = int(output_size_y - img_size_y % output_size_y)
        else:
            # For bigger images, is just the difference
            pad_y = output_size_y - img_size_y

        # Number of rows that need to be padded (top and bot)
        if not pad_x == output_size_x:
            pad_top = int(pad_x // 2)
            pad_bot = int(pad_x // 2)

            # if padding is not equally divided, pad +1 row to the top
            if not pad_x % 2 == 0:
                pad_top += 1
        else:
            pad_top = 0
            pad_bot = 0

        # Number of rows that need to be padded (left and right)
        if not pad_y == output_size_y:
            pad_left = int(pad_y // 2)
            pad_right = int(pad_y // 2)

            # if padding is not equally divided, pad +1 row to the left
            if not pad_y % 2 == 0:
                pad_left += 1
        else:
            pad_left = 0
            pad_right = 0

        return pad_top, pad_bot, pad_left, pad_right

    def get_medians(self, netcdf: netCDF4.Dataset, start_bin: int, window: int) -> np.ndarray:
        """
        netcdf (.nc): NETCDF4 Dataset.
        Opens requested band in received netcdf4 and aggregates based on given frequency
        """
        # Grab year from netcdf4's global attribute
        year = netcdf.patch_year

        # output intervals
        date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=self.group_freq)

        # out, aggregated array
        medians = np.empty((self.num_bands, window, self.patch_height, self.patch_width), dtype=self.medians_dtype)

        for band_id, band in enumerate(self.bands):

            # Load band data
            band_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf[band]))

            # Aggregate into time bins
            band_data = band_data.groupby_bins(
                'time',
                bins=date_range,
                right=True,
                include_lowest=False,
                labels=date_range[:-1]
            ).median(dim='time')

            # Upsample so months without data are initiated with NaN values
            band_data = band_data.resample(time_bins=self.group_freq).median(dim='time_bins')

            # Fill:
            # NaN months with linear interpolation
            # NaN months outsize of range (e.x month 12) using extrapolation
            band_data = band_data.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

            # Keep values within requested time window
            band_data = band_data.isel(time_bins=slice(start_bin, start_bin + window))

            # Convert to numpy array
            band_data = band_data[f'{band}'].values

            # If expand ratio is 1, that means current band has the same resolution as reference band
            expand_ratio = int(BANDS[band] / BANDS[REFERENCE_BAND])

            # If resolution does not match reference band, stretch it
            if expand_ratio != 1:
                band_data = np.repeat(band_data, expand_ratio, axis=1)
                band_data = np.repeat(band_data, expand_ratio, axis=2)

            medians[band_id, :, :, :] = np.expand_dims(band_data, axis=0)

        # Reshape so window length is first
        return medians.transpose(1, 0, 2, 3)


    def get_labels(self, netcdf: netCDF4.Dataset, start_bin: int) -> np.ndarray:

        # We should definitely explore how to produce labels for given binning frequency
        # labels can be timeseries, for the time being we only have 1 label per year
        # so just load and return that

        # Load and Convert to numpy array
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['labels']))['labels'].values

        return labels


    def load_medians(self, path: Path, subpatch_id: int, start_bin: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads precomputed medians for requested path.
        Medians are already padded and aggregated, so no need for further processing.
        Just load and return
        """
        # `medians` is a 4d numpy array (window length, bands, img_size, img_size)
        if self.fixed_window:
            medians = np.empty((6, self.num_bands, self.output_size[0], self.output_size[1]),
                                dtype=self.medians_dtype)
        else:
            medians = np.empty((self.window_len, self.num_bands, self.output_size[0], self.output_size[1]),
                                dtype=self.medians_dtype)

        padded_id = f'{str(subpatch_id).rjust(len(str(self.num_subpatches)), "0")}'

        median_files = sorted(path.glob(f'sub{padded_id}_bin*'))

        if self.fixed_window:
            start_month = 3
            end_month = 9
        else:
            start_month = start_bin
            end_month = start_bin + self.window_len

        for i, bin_idx in enumerate(range(start_month, end_month)):
            median = np.load(median_files[bin_idx]).astype(self.medians_dtype)
            medians[i] = median.copy()

        # Read labels
        labels = np.load(path / f'labels_sub{padded_id}.npy').astype(self.label_dtype)

        return medians, labels


    def get_window(self, idx: int) -> Tuple[int, int, int]:
        '''
        Returns the subpatch id, patch id and starting bin corresponding to
        the given index.

        For example, if we have 3 patches: 'patch0', 'patch1', 'patch2',
        each patch divided into 4 subpatches, 5 time bins in total
        and a time window with length 3, then indexing is performed as follows:
            - 0: patch0 - subpatch0 - bins [0, 1, 2]
            - 1: patch0 - subpatch1 - bins [0, 1, 2]
            - 2: patch0 - subpatch2 - bins [0, 1, 2]
            - 3: patch0 - subpatch3 - bins [0, 1, 2]
            - 4: patch1 - subpatch0 - bins [0, 1, 2]
            - ...
            - 12: patch0 - subpatch0 - bins [1, 2, 3]
            - 13: patch0 - subpatch1 - bins [1, 2, 3]
            - ...

        Arguments
        ---------
        idx: integer
            The index of the sequence.

        Returns
        -------
        (int, int, int): the starting time bin, the patch id and the subpatch id.
        '''
        start_bin = idx // (self.num_patches * self.num_subpatches)
        patch_id = (idx // self.num_subpatches) % self.num_patches
        subpatch_id = idx % self.num_subpatches

        return int(start_bin), int(patch_id), int(subpatch_id)


    def __getitem__(self, idx: int) -> dict:
        # The data item index (`idx`) corresponds to a single sequence.
        # In order to fetch the correct sequence, we must determine exactly which
        # patch, subpatch and bins it corresponds to.
        start_bin, patch_id, subpatch_id = self.get_window(idx)

        patch_id = self.patch_ids[patch_id]

        if self.saved_medians:
            # They are already computed, therefore we just load them
            block_dir = Path(self.medians_dir) / str(patch_id)

            # Read medians in time window
            medians, labels = self.load_medians(block_dir, subpatch_id, start_bin)
        else:
            # Find patch in COCO file
            patch = self.root_path_netcdf / self.coco.loadImgs(patch_id)[0]['file_name']

            # Load patch netcdf4
            patch_netcdf = netCDF4.Dataset(patch, 'r')

            # Compute on the fly each time, adds overhead for small output_size!!!
            # medians is a 4d numpy array (window length, bands, img_size, img_size)
            medians = self.get_medians(netcdf=patch_netcdf, start_bin=start_bin, window=self.window_len)

            # labels is a 3d numpy array (window length, img_size, img_size)
            # for the time being, we have yearly labels, so window_len will always be 1
            labels = self.get_labels(netcdf=patch_netcdf, start_bin=start_bin)

            if self.requires_pad:
                medians = np.pad(medians,
                                 pad_width=((0, 0), (0, 0), (self.pad_top, self.pad_bot), (self.pad_left, self.pad_right)),
                                 mode='constant',
                                 constant_values=0)

                labels = np.pad(labels,
                                pad_width=((self.pad_top, self.pad_bot), (self.pad_left, self.pad_right)),
                                mode='constant',
                                constant_values=0
                                )

            if self.requires_subpatching:
                window_len, num_bands, width, height = medians.shape

                # Side_h should be equal length of side_w
                side_h = self.output_size[0]
                side_w = self.output_size[1]
                num_subpatches_h = int(self.padded_patch_height // side_h)
                num_subpatches_w = int(self.padded_patch_width // side_w)

                # Reshape medians
                # From:             (window length, bands, pad_img_size, pad_img_size)
                # To                (window length, bands, N, output_shape[0], M, output_shape[1])
                # Transpose         (N, M, window length, bands, output_shape[0], output_shape[1])
                # Reshape           (N * M, window length, bands, output_shape[0], output_shape[1])
                medians = medians.reshape(window_len, num_bands, num_subpatches_w, side_w, num_subpatches_h, side_h) \
                    .transpose(2, 4, 0, 1, 3, 5) \
                    .reshape(-1, window_len, num_bands, side_w, side_h)

                # Same for labels, but no bands and window length dimensions
                labels = labels.reshape(num_subpatches_w, side_w, num_subpatches_h, side_h)\
                    .transpose(0, 2, 1, 3)\
                    .reshape(-1, side_w, side_h)

                # Return requested sub-patch
                medians = medians[subpatch_id]
                labels = labels[subpatch_id]

        # Normalize data to range [0-1]
        if self.requires_norm:
            medians = np.divide(medians, NORMALIZATION_DIV)

        if self.window_len == 1:
            # Remove window_len dimension
            medians = medians.squeeze(axis=0)

        out = {}

        if self.return_parcels:
            parcels = labels != 0
            out['parcels'] = parcels

        if self.binary_labels:
            # Map 0: background class, 1: parcel
            labels[labels != 0] = 1
        else:
            # Map labels to 0-len(unique(crop_id)) see config
            # labels = np.vectorize(self.linear_encoder.get)(labels)
            _ = np.zeros_like(labels)
            for crop_id, linear_id in self.linear_encoder.items():
                _[labels == crop_id] = linear_id
            labels = _

        # Map all classes NOT in linear encoder's values to 0
        labels[~np.isin(labels, list(self.linear_encoder.values()))] = 0

        out['medians'] = medians.astype(self.medians_dtype)
        out['labels'] = labels.astype(self.label_dtype)
        out['idx'] = idx

        if self.return_masks:
            out['masks'] = hollstein_mask(out['medians'],
                                          clouds=self.masks['clouds'],
                                          cirrus=self.masks['cirrus'],
                                          shadows=self.masks['shadow'],
                                          snow=self.masks['snow'],
                                          requires_norm=self.requires_norm,
                                          reference_bands=self.bands)

        return out


    def __len__(self):
        '''
        Computes the total number of produced sequences,
        i.e. one item will contain the patch medians for a single time window
        '''
        return int(self.num_patches * self.num_subpatches) * (self.num_buckets - self.window_len + 1)
