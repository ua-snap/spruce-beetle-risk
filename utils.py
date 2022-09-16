"""Some utilities for assisting with the data pipeline"""

import subprocess
import numpy as np


def clip_with_gdal(src_fp, cut_fp):
    """"""
    clipped_fp = src_fp.parent.joinpath(src_fp.name.replace(".tif", "_clip.tif"))
    clipped_fp.unlink(missing_ok=True)
    _ = subprocess.call(
        [
            "gdalwarp",
            "-cutline",
            cut_fp,
            "-crop_to_cutline",
            "-q",
            src_fp,
            clipped_fp,
        ]
    )
    return clipped_fp
