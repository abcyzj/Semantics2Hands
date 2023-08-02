import os
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch3d.transforms import standardize_quaternion

from data.motion_dataset import MotionData


class MixedData0(Dataset):
    """
    Mixed data for many skeletons but one topologies
    """
    def __init__(self, motions, skeleton_idx, data_augment: bool):
        super(MixedData0, self).__init__()

        self.motions = motions
        self.motions_reverse = torch.tensor(self.motions.numpy()[..., ::-1].copy())
        self.skeleton_idx = skeleton_idx
        self.length = motions.shape[0]
        self.data_augment = data_augment

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if not self.data_augment or torch.rand(1) < 0.5:
            return [self.motions[item], self.skeleton_idx[item]]
        else:
            return [self.motions_reverse[item], self.skeleton_idx[item]]


class MixedData(Dataset):
    """
    data_gruop_num * 2 * samples
    """
    def __init__(self, data_path: str, data_augment: bool, window_size: int, normalization: bool, datasets_groups: List[List[str]]):
        self.data_augment = data_augment
        self.windwo_size = window_size
        self.normalization = normalization
        self.datasets_groups = datasets_groups
        self.final_data = []
        self.length = 0
        self.means = []
        self.vars = []
        dataset_num = 0
        total_length = 10000000
        all_datas = []
        for datasets in datasets_groups:
            means_group = []
            vars_group = []
            dataset_num += len(datasets)
            tmp = []
            for i, dataset in enumerate(datasets):
                tmp.append(MotionData(data_path, dataset, False, window_size, normalization))

                mean = np.load(os.path.join(data_path, 'mean_var', '{}_mean.npy'.format(dataset)))
                var = np.load(os.path.join(data_path, 'mean_var', '{}_var.npy'.format(dataset)))
                mean = torch.tensor(mean)
                mean = mean.reshape((1,) + mean.shape)
                var = torch.tensor(var)
                var = var.reshape((1,) + var.shape)

                means_group.append(mean)
                vars_group.append(var)

                total_length = min(total_length, len(tmp[-1]))
            all_datas.append(tmp)
            means_group = torch.cat(means_group, dim=0)
            vars_group = torch.cat(vars_group, dim=0)
            self.means.append(means_group)
            self.vars.append(vars_group)

        for datasets in all_datas:
            pt = 0
            motions = []
            skeleton_idx = []
            for dataset in datasets:
                motions.append(dataset[:])
                skeleton_idx += [pt] * len(dataset)
                pt += 1
            motions = torch.cat(motions, dim=0)
            if self.length != 0 and self.length != len(skeleton_idx):
                self.length = min(self.length, len(skeleton_idx))
            else:
                self.length = len(skeleton_idx)
            self.final_data.append(MixedData0(motions, skeleton_idx, data_augment))

    def denorm(self, gid, pid, data):
        means = self.means[gid][pid, ...]
        var = self.vars[gid][pid, ...]
        return data * var + means

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        res = []
        for data in self.final_data:
            res.append(data[item])
        return res

    @staticmethod
    def resolve_tensor(batch: List[torch.Tensor]):
        res = []
        for motion, character_idx in batch:
            if motion.shape[1] == (10 + 180 + 84 + 60 + 1200 + 540): # mano
                B, C, T = motion.shape
                shapes = motion[:, 0:10, :].reshape(B, 10, T)
                m_tbs = motion[:, 10:190, :].reshape(B, 5, 4, 3, 3, T)
                rotations = motion[:, 190:274, :].reshape(B, 21, 4, T)
                bs_rotations = motion[:, 274:334, :].reshape(B, 5, 3, 4, T)
                rotations = standardize_quaternion(rotations.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                bs_rotations = standardize_quaternion(bs_rotations.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
                relative_finger_tbs = motion[:, 334:1534, :].reshape(B, 20, 20, 3, T)
                relative_palm_tbs = motion[:, 1534:2074, :].reshape(B, 20, 9, 3, T)
                res.append({
                    'shapes': shapes,
                    'm_tbs': m_tbs,
                    'rotations': rotations,
                    'bs_rotations': bs_rotations,
                    'relative_finger_tbs': relative_finger_tbs,
                    'relative_palm_tbs': relative_palm_tbs,
                    'character_idx': character_idx,
                })
            elif motion.shape[1] == (63 + 180 + 84 + 60 + 1200 + 540): # mixamo
                B, C, T = motion.shape
                offset = motion[:, 0:63, :].reshape(B, 21, 3, T)
                offset[:, 0] = 0.0
                m_tbs = motion[:, 63:243, :].reshape(B, 5, 4, 3, 3, T)
                rotations = motion[:, 243:327, :].reshape(B, 21, 4, T)
                bs_rotations = motion[:, 327:387, :].reshape(B, 5, 3, 4, T)
                rotations = standardize_quaternion(rotations.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                bs_rotations = standardize_quaternion(bs_rotations.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
                relative_finger_tbs = motion[:, 387:1587, :].reshape(B, 20, 20, 3, T)
                relative_palm_tbs = motion[:, 1587:2127, :].reshape(B, 20, 9, 3, T)
                res.append({
                    'offset': offset,
                    'm_tbs': m_tbs,
                    'rotations': rotations,
                    'bs_rotations': bs_rotations,
                    'relative_finger_tbs': relative_finger_tbs,
                    'relative_palm_tbs': relative_palm_tbs,
                    'character_idx': character_idx
                })
            else:
                raise NotImplementedError('Unknown motion shape: {}'.format(motion.shape))

        return res


class MixedDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, data_augment: bool, window_size: int, normalization: bool, batch_size: int, num_workers: int):
        super().__init__()
        self.data_path = data_path
        self.data_augment = data_augment
        self.window_size = window_size
        self.normalization = normalization
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        train_characters = [
            ['Aj', 'BigVegas', 'Kaya'] * 9, # for equal each frames in each group
            ['Mano0', 'Mano1', 'Mano2', 'Mano3', 'Mano4', 'Mano5', 'Mano6']
        ]
        val_characters = [
            ['SportyGranny'],
            ['Mano7', 'Mano8', 'Mano9']
        ]
        test_characters = [
            ['SportyGranny'] * 9,
            ['Mano7', 'Mano8', 'Mano9']
        ]

        # Uncomment this to perform intra-domain test
        # test_characters = [
        #     ['Kaya'],
        #     ['SportyGranny']
        # ]


        if stage == 'fit':
            self.mixdata_train = MixedData(self.data_path, self.data_augment, self.window_size, self.normalization, train_characters)

        if stage == 'validate' or stage == 'fit':
            self.mixdata_val = MixedData(self.data_path, False, self.window_size, self.normalization, val_characters)

        if stage == 'test':
            self.mixdata_test = MixedData(self.data_path, False, self.window_size, self.normalization, test_characters)

    def train_dataloader(self):
        return DataLoader(self.mixdata_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.mixdata_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.mixdata_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
