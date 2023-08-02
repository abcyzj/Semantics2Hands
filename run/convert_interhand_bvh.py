import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import smplx
import torch
import trimesh
from pytorch3d.transforms import axis_angle_to_quaternion
from tqdm import tqdm
from trimesh import Trimesh
from trimesh.exchange.obj import export_obj

from utils.armatures import BaseArmature, MANOArmature
from utils.mesh import load_path_from_axis
import utils.BVH as BVH
from utils.Animation import Animation
from utils.Quaternions_old import Quaternions


def mano_pose_to_quaternion(mano_model: smplx.MANO, pose: List[float]):
    pose = torch.FloatTensor(pose).view(-1, 48)
    pose = pose + mano_model.pose_mean
    pose = pose.view(-1, 16, 3)
    pose = axis_angle_to_quaternion(pose) # (B, 16, 4)
    finger_tip_pose = torch.zeros(pose.shape[0], 5, 4, device=pose.device, dtype=pose.dtype) # (B, 5, 4)
    finger_tip_pose[..., 0] = 1.0
    pose = torch.cat([pose, finger_tip_pose], dim=1) # (B, 21, 4)
    return pose


def display_armature(armature: BaseArmature):
    tbs_matrix = armature.tbs_matrix_global()
    joints = armature.joints.detach().cpu()[0, 0].numpy().squeeze()
    factor = 1 / np.linalg.norm(joints[0] - joints[1])
    mesh = Trimesh(vertices=armature.verts.detach().cpu()[0, 0].numpy().squeeze() * factor, faces=armature.faces)
    joints = joints * factor
    paths = []
    for i in range(20):
        x_path = load_path_from_axis(tbs_matrix[..., i//4, i%4, :, 0].detach().cpu()[0, 0].numpy().squeeze(), joints[i+1], color=np.array([255, 0, 0, 255]), n_points=3)
        y_path = load_path_from_axis(tbs_matrix[..., i//4, i%4, :, 1].detach().cpu()[0, 0].numpy().squeeze(), joints[i+1], color=np.array([0, 255, 0, 255]), n_points=3)
        z_path = load_path_from_axis(tbs_matrix[..., i//4, i%4, :, 2].detach().cpu()[0, 0].numpy().squeeze(), joints[i+1], color=np.array([0, 0, 255, 255]), n_points=3)
        paths.extend([x_path, y_path, z_path])
    trimesh.Scene([mesh] + paths).show()


def export_armature(armature: BaseArmature, p: str):
    mesh = Trimesh(vertices=armature.verts.detach().cpu().numpy().squeeze(), faces=armature.faces)
    with open(p, 'w') as f:
        f.write(export_obj(mesh))


def main(args: Namespace):
    mano_layer: Dict[str, smplx.MANO] = {'right': smplx.create(args.smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(args.smplx_path, 'mano', use_pca=False, is_rhand=False)}
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
        # print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:,0,:] *= -1

    with open(args.input, 'r') as f:
        train_mano_data = json.load(f)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mano_axis_data, 'rb') as f:
        mano_axis_data = pickle.load(f)

    hand_key = 'right'
    is_rhand = (hand_key == 'right') # Only get right hand data

    for capture_id in train_mano_data.keys():
        mano_model = smplx.create(args.smplx_path, 'mano', use_pca=False, is_rhand=is_rhand)
        frame_ids = [int(k) for k in train_mano_data[capture_id].keys()]
        frame_ids.sort()
        ids = np.asarray(frame_ids)
        mean_interval = np.mean(ids[1:] - ids[:-1])
        if mean_interval > 50:
            continue

        frame_interval = frame_ids[1] - frame_ids[0]
        if frame_interval != frame_ids[2] - frame_ids[1]:
            raise ValueError('Invalid frame interval')
        sequences = []
        cur_seq = []
        for i, frame_id in enumerate(frame_ids):
            if i == 0:
                cur_seq.append(frame_id)
            elif frame_id - frame_ids[i-1] > frame_interval or train_mano_data[capture_id][str(frame_id)][hand_key] is None:
                if len(cur_seq) > 1:
                    sequences.append(cur_seq)
                cur_seq = [frame_id] if train_mano_data[capture_id][str(frame_id)][hand_key] is not None else []
            else:
                cur_seq.append(frame_id)

        def get_seq_data(seq):
            shapes = []
            hand_rotations = []
            for frame_id in seq:
                mano_data = train_mano_data[capture_id][str(frame_id)]
                if mano_data[hand_key] is None:
                    raise ValueError('Invalid hand data')
                hand_rotations.append(mano_pose_to_quaternion(mano_model, mano_data[hand_key]['pose']))
                shape = torch.FloatTensor(mano_data[hand_key]['shape']).view(1, -1)
                shapes.append(shape)

            hand_rotations = torch.cat(hand_rotations, dim=0)
            shapes = torch.cat(shapes, dim=0)
            armature = MANOArmature(is_rhand=is_rhand, smplx_path=args.smplx_path, finger_axes=mano_axis_data, shape=shapes)
            hand_rotations = armature.get_hand_data(hand_rotations.unsqueeze(1))
            armature.hand_rotations = hand_rotations
            return armature.hand_rotations.squeeze(1).cpu().numpy(), armature.offsets.mean(dim=0).cpu().numpy(), armature.hand_j_labels, armature.hand_parents, shapes

        for seq in tqdm(sequences, desc=f'Processing {capture_id}'):
            rotations, offsets, joint_names, parents, shapes = get_seq_data(seq)
            F, J = rotations.shape[:2]
            orients = np.zeros((J, 4))
            orients[..., 0] = 1.0
            orients = Quaternions(orients)
            anim = Animation(Quaternions(rotations), np.zeros((F, J, 3)), orients, offsets * 100, parents)
            out_bvh_f = output_dir / f'{capture_id}_{seq[0]}_{seq[-1]}.bvh'
            BVH.save(out_bvh_f.as_posix(), anim, joint_names, 1/5)
            out_shape_f = output_dir / f'{capture_id}_{seq[0]}_{seq[-1]}.npy'
            np.save(out_shape_f.as_posix(), shapes.cpu().numpy())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--smplx_path', type=str, default='artifact/smplx/models')
    parser.add_argument('--input', type=str, default='artifact/InterHand2.6M/annotations/train/InterHand2.6M_train_MANO_NeuralAnnot.json')
    parser.add_argument('--mano_axis_data', type=str, default='artifact/MixHand/finger_data/mano_finger_axis.pkl')
    parser.add_argument('--output_dir', type=str, default='artifact/InterHand2.6M/annotations/train/bvh')
    args = parser.parse_args()

    main(args)
