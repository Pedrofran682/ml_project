from collections import namedtuple
import glob
import os
import functools
import csv
import numpy as np
import collections
import diskcache
from .Ct import CT
from .config import DATA_FOLDER_PATH


CandidateInfoTuple = namedtuple('CandidateInfoTuple', 
                                'isNodule_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob(os.path.join(DATA_FOLDER_PATH, 'subset*/*.mhd'))
    presentOnDisk_set = {os.path.split(path)[-1][:-4] for path in mhd_list}

    diameter_dict = {}
    file_path = os.path.join(DATA_FOLDER_PATH, 'annotations.csv')
    print(f"Reading {file_path}")
    with open(file_path, 'r') as file:
        for row in list(csv.reader(file))[1:]:
            series_uid: str = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
                (annotation_center_xyz, annotation_diameter_mm)
            )
    file.close()

    candidateInfo_list = []
    file_path = os.path.join(DATA_FOLDER_PATH, 'candidates.csv')
    print(f"Reading {file_path}")
    with open(file_path, 'r') as file:
        for row in list(csv.reader(file))[1:]:
            series_uid: str = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup
                for index in range(3):
                    delta_mm = abs(candidateCenter_xyz[index] -
                                   annotation_center_xyz[index])
# Divides the diameter by 2 to get the radius, and divides
# the radius by 2 to require that the two nodule center
# points not be too far apart relative to the size of the nodule.
                    if delta_mm > annotation_diameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotation_diameter_mm
                    break
            candidateInfo_list.append(
                CandidateInfoTuple(isNodule_bool, candidateDiameter_mm,
                                   series_uid, candidateCenter_xyz))
    file.close()
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1, typed=True)
def getCt(series_uid: str):
    print(f"{series_uid = }")
    return CT(series_uid)


def getCtRawCandidate(series_uid: str, center_xyz: float, width_irc: int):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.GetRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc





