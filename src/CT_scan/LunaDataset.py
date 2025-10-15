import torch
import copy
from .dsets import getCandidateInfoList, getCtRawCandidate
from torch.utils.data import Dataset
import logging

log = logging.getLogger(__name__)

class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool: bool =None,series_uid: str =None):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx: int):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(candidateInfo_tup.series_uid,
                                                 candidateInfo_tup.center_xyz,
                                                 width_irc)
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32).unsqueeze(0)
        post_t = torch.tensor([not candidateInfo_tup.isNodule_bool, candidateInfo_tup.isNodule_bool],
                              dtype=torch.long)

        # print(f"{candidateInfo_tup.series_uid = }  ")
        # print(f"{torch.tensor(center_irc).shape = } ")
        # print(f"{post_t.shape = } ")
        # print(f"{candidate_t.shape = } ")

        return (candidate_t, post_t, candidateInfo_tup.series_uid, torch.tensor(center_irc))


