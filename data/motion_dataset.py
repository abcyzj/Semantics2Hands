import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.Quaternions import Quaternions


class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, data_path: str, name: str, data_augment: bool, window_size: int, normalization: bool):
        super(MotionData, self).__init__()
        file_path = os.path.join(data_path, name + '.npy')
        self.name = name
        self.data_augment = data_augment
        self.window_size = window_size

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)
        motions = list(motions)
        new_windows = self.get_windows(motions)
        self.data.append(new_windows)
        self.data = torch.cat(self.data)
        self.data = self.data.permute(0, 2, 1)

        if normalization:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.var = torch.var(self.data, (0, 2), keepdim=True)
            self.var = self.var ** (1/2)
            idx = self.var < 1e-5
            self.var[idx] = 1
            self.data = (self.data - self.mean) / self.var
        else:
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.mean.zero_()
            self.var = torch.ones_like(self.mean)

        self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())

        self.reset_length_flag = 0
        self.virtual_length = 0
        print('Window count: {}, total frame (without downsampling): {}'.format(len(self), self.total_frame))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
        if not self.data_augment or np.random.randint(0, 2) == 0:
            return self.data[item]
        else:
            return self.data_reverse[item]

    def get_windows(self, motions):
        new_windows = []

        for motion in motions:
            self.total_frame += motion.shape[0]
            motion = self.subsample(motion)
            self.motion_length.append(motion.shape[0])
            step_size = self.window_size // 2
            window_size = step_size * 2
            n_window = motion.shape[0] // step_size - 1
            for i in range(n_window):
                begin = i * step_size
                end = begin + window_size

                new = motion[begin:end, :]

                new = new[np.newaxis, ...]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)

    def subsample(self, motion):
        if self.name[:4] == 'Mano': # For Mano, do not subsample
            return motion
        else:
            return motion[::12, :] # For mixamo, subsample from 60fps to 5fps

    def denormalize(self, motion):
        if self.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans
