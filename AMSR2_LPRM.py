"""
AMSR2_LPRM.py: A module for processing GCOM-W1 satellite AMSR2 LPRM-based soil moisture data.

Shared file-listing/array-stacking logic lives in readers_common; this module
keeps the AMSR2-specific helpers (shape correction, bit masking).
"""
import numpy as np

from . import readers_common


def extract_filelist_doy(directory, year):
    """List .nc4 files and their DOY from 'yyyy.mm.dd' subdirectories."""
    return readers_common.extract_filelist_doy(directory, year, ext='nc4')


def create_array_from_nc(file_list, data_doy, year, variable_name):
    """Stack a variable from daily .nc4 files into a (x, y, doy) array."""
    return readers_common.create_array_from_nc(file_list, data_doy, year, variable_name)


def correct_shape(data):
    data = np.fliplr(data.transpose((1, 0, 2)))

    return data


# bit meaning
# 1 Negative optical depth in X
# 2 Negative optical depth in C2
# 3 Negative optical depth in C1
# 4 High optical depth value in X
# 5 High optical depth value in C2
# 6 High optical depth value in C1
# 7 No valid data
# 8 Ice
# 9 Not processed
def create_mask(bit_mask_3d, bit_position=8):
    # Ensure the bit mask is in an integer format
    bit_mask_3d = bit_mask_3d.astype(np.int32)

    # Identify where the requested bit (default: Ice) is set
    ice_mask = (bit_mask_3d & (1 << bit_position)) != 0

    # Convert boolean mask to integer mask (1 for ice, 0 for no ice)
    ice_mask = ice_mask.astype(int)

    return ice_mask
