from .util import XyzTuple, xyz2irc, irc2xyz
import SimpleITK as sitk
import glob
import os
import numpy as np
DATA_FOLDER_PATH = "data/CT_scan/LUNA"

class CT:
    def __init__(self, series_uid: str):
        mhd_path = glob.glob(os.path.join(DATA_FOLDER_PATH,
                                          f'subset*/{series_uid}.mhd'))[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)
        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def GetRawCandidate(self, center_xyz: float, width_irc: float):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)
        slice_list = []
        for axis, certer_val in enumerate(center_irc):
            start_ndx = int(round(certer_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])
            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc
