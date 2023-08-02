import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.bvh_parser import BVH_file
from data.motion_dataset import MotionData
from utils.armatures import MANOArmature, MixamoArmature
from utils.armature_config import config_mixamo_armature


def collect_bvh(data_path, smplx_path, character, files, is_rhand: bool = True):
    print('begin {}'.format(character))
    is_mano = (character[:4].lower() == 'mano')
    if is_mano:
        with open(os.path.join(data_path, 'finger_data', 'mano_finger_axis.pkl'), 'rb') as f:
            finger_axes = pickle.load(f)
            mesh_data = None
    else:
        with open(os.path.join(data_path, 'finger_data', f'{character}_finger_axis.pkl'), 'rb') as f:
            finger_axes = pickle.load(f)
        mesh_data = np.load(os.path.join(data_path, 'finger_data', f'{character}_mesh_data.npz'))
    motions = []


    for i, motion in enumerate(tqdm(files, desc=character)):
        if not os.path.exists(os.path.join(data_path, character, motion)):
            continue
        file = BVH_file(os.path.join(data_path, character, motion))
        if is_mano:
            shape_f = os.path.join(data_path, character, motion.split('.')[0] + '.npy')
            if os.path.exists(shape_f):
                shapes = torch.from_numpy(np.load(shape_f))
            else:
                F = file.anim.rotations.shape[0]
                shapes = torch.zeros(F, 10) # if shape is not provided, use mean shape
            armature = MANOArmature(is_rhand, smplx_path, finger_axes, shapes)
        else:
            armature = MixamoArmature(is_rhand, finger_axes, mesh_data)
        new_motion = file.to_tensor(armature).permute((1, 0)).numpy()
        motions.append(new_motion)

    save_file = data_path + character + '.npy'

    np.save(save_file, np.asarray(motions, dtype=object))
    print('Npy file saved at {}'.format(save_file))


def write_statistics(data_path: str, character: str, window_size: int):
    dataset = MotionData(data_path, character, False, window_size, True)
    path = os.path.join(data_path, 'mean_var')

    mean = dataset.mean
    var = dataset.var
    mean = mean.cpu().numpy()[0, ...]
    var = var.cpu().numpy()[0, ...]

    np.save(os.path.join(path, '{}_var.npy'.format(character)), var)
    np.save(os.path.join(path, '{}_mean.npy'.format(character)), mean)


if __name__ == '__main__':
    config_mixamo_armature()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='artifact/MixHand/')
    parser.add_argument('--smplx_path', type=str, default='artifact/smplx/models')
    parser.add_argument('--is_rhand', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=8)
    args = parser.parse_args()
    characters = [f for f in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, f))]
    if 'std_bvhs' in characters: characters.remove('std_bvhs')
    if 'mean_var' in characters: characters.remove('mean_var')
    if 'ref' in characters: characters.remove('ref')
    if 'finger_data' in characters: characters.remove('finger_data')

    Path(os.path.join(args.data_path, 'std_bvhs')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.data_path, 'mean_var')).mkdir(parents=True, exist_ok=True)

    is_rhand = (args.is_rhand == 1)
    for character in characters:
        data_path = os.path.join(args.data_path, character)
        files = sorted([f for f in os.listdir(data_path) if f.endswith(".bvh")])

        collect_bvh(args.data_path, args.smplx_path, character, files, is_rhand)
        write_statistics(args.data_path, character, args.window_size)
