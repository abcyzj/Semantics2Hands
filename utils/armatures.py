import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pyvista as pv
import smplx
import torch
import trimesh
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  matrix_to_quaternion, quaternion_invert,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle,
                                  quaternion_to_matrix, standardize_quaternion)
from smplx.joint_names import SMPLH_JOINT_NAMES
from trimesh import Trimesh
from trimesh.collision import CollisionManager
from trimesh.exchange.obj import export_obj

import utils.BVH as BVH
from model.kinematics import ForwardKinematics
from utils.lbs import SkinnableMesh
from utils.mesh import (bary_to_axis_batch, extract_joint_mesh,
                        load_path_from_axis)


class BaseArmature:
    def __init__(self, joint_names: List[str], parents: List[int], hand_labels: List[List], finger_axes: torch.Tensor, is_rhand: bool):
        self.joint_names = joint_names
        self.is_rhand = is_rhand

        single_hand_labels = hand_labels[0] if is_rhand else hand_labels[1]
        self.hand_j_labels = [single_hand_labels[0]]
        if len(single_hand_labels) == 7:
            self.hand_j_labels.extend(single_hand_labels[6]) # Add palm joints
        for finger_labels in single_hand_labels[1:6]:
            self.hand_j_labels.extend(finger_labels)
        self.finger_indices_in_hand = []
        self.finger_ori_indices = []
        self.finger_names = []
        for finger_labels in single_hand_labels[1:6]:
            self.finger_indices_in_hand.append([self.hand_j_labels.index(j) for j in finger_labels])
            self.finger_ori_indices.append([joint_names.index(j) for j in finger_labels])
            self.finger_names.append(finger_labels)
        self.finger_indices_in_hand = np.array(self.finger_indices_in_hand)
        self.hand_j_ori_indices = [joint_names.index(j) for j in self.hand_j_labels]
        self.hand_panel_j_labels = [single_hand_labels[0], single_hand_labels[3][0], single_hand_labels[4][0]]
        self.hand_panel_j_indices = [self.hand_j_labels.index(j) for j in self.hand_panel_j_labels]

        self.ori2hand = {}
        self.hand2ori = {}
        self.hand_parents = []
        for hand_idx, ori_idx in enumerate(self.hand_j_ori_indices):
            self.ori2hand[ori_idx] = hand_idx
            self.hand2ori[hand_idx] = ori_idx
            self.hand_parents.append(parents[ori_idx])
        for hand_idx, p in enumerate(self.hand_parents):
            if p not in self.ori2hand and hand_idx == 0:
                self.hand_parents[hand_idx] = -1
                continue
            while p not in self.ori2hand:
                p = parents[p]
            self.hand_parents[hand_idx] = self.ori2hand[p]
        self.hand_parents = np.asarray(self.hand_parents)
        self.fk = ForwardKinematics(self.hand_parents)

        if finger_axes is not None:
            self.m_tbs = finger_axes.transpose(-1, -2)
            # uncomment to see how tbs coordiante works
            # self.m_tbs = torch.eye(3, dtype=self.m_tbs.dtype, device=self.m_tbs.device).unsqueeze(0).repeat(1, 5, 4, 1, 1)
        else:
            self.m_tbs = None

        self.rest_pose = None
        self.hand_offsets = None
        self._hand_rotations = None
        self._bs_rotations = None


    def build_finger_frame(self, offsets: torch.Tensor):
        '''
        offsets: (B, J_hand, 3)
        '''
        B, J = offsets.shape[:2]
        ori_rest_pose = torch.zeros(B, 1, J, 4, dtype=offsets.dtype, device=offsets.device)
        ori_rest_pose[..., 0] = 1.0
        root_positions = torch.zeros(B, 1, 3, dtype=offsets.dtype, device=offsets.device)
        ori_rest_pose_global, ori_rest_positions = self.fk.forward(ori_rest_pose, root_positions, offsets, ret_rot=True, ret_ori_repr=True)
        rest_hand_panel = ori_rest_positions[..., self.hand_panel_j_indices, :] # (B, 1, 3, 3)
        n_hand_panel = torch.cross(rest_hand_panel[..., 2, :] - rest_hand_panel[..., 0, :], rest_hand_panel[..., 1, :] - rest_hand_panel[..., 0, :])
        if self.is_rhand:
            n_hand_panel = -n_hand_panel # make sure the normal is pointing to the sky
        n_hand_panel = n_hand_panel / torch.norm(n_hand_panel, dim=2, keepdim=True)

        rest_pose = ori_rest_pose.clone()
        rest_positions = ori_rest_positions.clone()
        rest_pose_global = ori_rest_pose_global.clone()
        hand_middle_dir = rest_positions[..., self.finger_indices_in_hand[0, 0], :] - rest_positions[..., 0, :] # direction from wrist to middle finger
        hand_middle_dir = hand_middle_dir - torch.linalg.vecdot(hand_middle_dir, n_hand_panel).unsqueeze(-1) * n_hand_panel # Make sure hand_middle_dir is on the hand panel
        hand_middle_dir = hand_middle_dir / torch.norm(hand_middle_dir, dim=-1, keepdim=True)
        for finger_idx, finger_j_indices in enumerate(self.finger_indices_in_hand):
            for j in finger_j_indices[:3]: # ignore finger tip
                ori_bone_dir = rest_positions[..., j+1, :] - rest_positions[..., j, :]
                ori_bone_dir = ori_bone_dir / torch.norm(ori_bone_dir, dim=-1, keepdim=True)
                bone_dir = ori_bone_dir - torch.linalg.vecdot(ori_bone_dir, n_hand_panel).unsqueeze(-1) * n_hand_panel
                bone_dir = bone_dir / torch.norm(bone_dir, dim=-1, keepdim=True)
                bone_transform_axis = torch.cross(ori_bone_dir, bone_dir)
                bone_transform = bone_transform_axis / torch.norm(bone_transform_axis, dim=-1, keepdim=True) * torch.asin(torch.norm(bone_transform_axis, dim=-1, keepdim=True))
                bone_transform = axis_angle_to_quaternion(bone_transform)
                if finger_idx <= 3: # middle, ring, pinky, index
                    delta_transform_axis = torch.cross(bone_dir, hand_middle_dir)
                    delta_transform = delta_transform_axis / torch.norm(delta_transform_axis, dim=-1, keepdim=True) * torch.asin(torch.norm(delta_transform_axis, dim=-1, keepdim=True))
                    delta_transform = axis_angle_to_quaternion(delta_transform)
                    bone_transform = quaternion_multiply(delta_transform, bone_transform)
                bone_transform = quaternion_multiply(bone_transform, rest_pose_global[..., j, :])
                bone_transform = quaternion_multiply(quaternion_invert(rest_pose_global[..., j, :]), bone_transform)
                rest_pose[..., j, :] = bone_transform
                rest_pose_global, rest_positions = self.fk.forward(rest_pose, root_positions, offsets, ret_rot=True, ret_ori_repr=True)
                bone_dir = rest_positions[..., j+1, :] - rest_positions[..., j, :]

        self.rest_pose = rest_pose
        self.hand_offsets = offsets.clone()


    def hand_fk(self, hand_rotations: torch.Tensor, hand_offsets: torch.Tensor):
        '''
        hand_rotations: (B, T, J_hand, 4)
        hand_offsets: (B, J_hand, 3)
        return: (B, T, J_hand, 3)
        '''
        B, T, J = hand_rotations.shape[:3]
        root_positions = torch.zeros(B, J, 3, dtype=hand_offsets.dtype, device=hand_offsets.device)
        return self.fk.forward(hand_rotations, root_positions, hand_offsets)


    def get_hand_data(self, rotations: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        '''
        rotations: (B, T, J_body, 4)
        offsets: (B, T, J_body, 3)
        return: (B, T, J_hand, 4), (B, J_hand, 3)
        '''
        if offsets is not None:
            return rotations[..., self.hand_j_ori_indices, :], offsets[..., self.hand_j_ori_indices, :]
        else:
            return rotations[..., self.hand_j_ori_indices, :]


    def tbs_matrix_global(self, hand_rotations: Optional[torch.Tensor] = None):
        '''
        hand_rotations: (B, T, J_hand, 4)
        return: (B, T, 5, 4, 3, 3)
        '''
        assert self.hand_offsets is not None
        if hand_rotations is None:
            hand_rotations = self.hand_rotations
        B, T = hand_rotations.shape[:2]
        hand_rotations_global = self.fk.forward(hand_rotations, torch.zeros(B, T, 3, dtype=hand_rotations.dtype, device=hand_rotations.device), self.hand_offsets.to(hand_rotations.device), ret_rot=True, ret_ori_repr=True)[0]
        tbs_matrix = self.m_tbs.to(hand_rotations.device, hand_rotations.dtype).unsqueeze(1).repeat(1, T, 1, 1, 1, 1)
        tbs_matrix_global = quaternion_to_matrix(hand_rotations_global[..., self.finger_indices_in_hand, :]) @ tbs_matrix
        return tbs_matrix_global


    def relative_tbs_coordinates(self, normalized: bool = True, palm2thumb: bool = True):
        '''
        return: (B, T, 20, 20, 3)
        '''
        joints = self.joints # (B, T, J_hand, 3)
        B, T = joints.shape[:2]
        finger_joints = joints[..., self.finger_indices_in_hand, :].reshape(B, T, 20, 3)
        tbs_matrix_global = self.tbs_matrix_global().reshape(B, T, 20, 3, 3)
        tbs_matrix_global_inverted = tbs_matrix_global.transpose(-1, -2).unsqueeze(3) # (B, T, 20, 1, 3, 3)
        relative_finger_coordinates = (finger_joints.unsqueeze(2) - finger_joints.unsqueeze(3)).unsqueeze(-1) # (B, T, 20, 20, 3, 1)
        relative_finger_coordinates = torch.matmul(tbs_matrix_global_inverted, relative_finger_coordinates) # (B, T, 20, 20, 3, 1)
        relative_finger_coordinates = relative_finger_coordinates.squeeze(-1) # (B, T, 20, 20, 3)
        palm_anchors = self.palm_anchors # (B, T, 9, 3)
        relative_palm_coordinates = (palm_anchors.unsqueeze(2) - finger_joints.unsqueeze(3)).unsqueeze(-1) # (B, T, 20, 9, 3, 1)
        relative_palm_coordinates = torch.matmul(tbs_matrix_global_inverted, relative_palm_coordinates) # (B, T, 20, 9, 3, 1)
        relative_palm_coordinates = relative_palm_coordinates.squeeze(-1) # (B, T, 20, 9, 3)
        if not palm2thumb:
            relative_palm_coordinates = relative_palm_coordinates[..., :-4, :, :]
        if normalized:
            wrist_to_middle = (finger_joints[..., 0, :] - joints[..., 0, :]).unsqueeze(2).unsqueeze(3).norm(dim=-1, keepdim=True) # (B, T, 1, 1, 1)
            relative_finger_coordinates = relative_finger_coordinates / wrist_to_middle
            relative_palm_coordinates = relative_palm_coordinates / wrist_to_middle
        return relative_finger_coordinates, relative_palm_coordinates


    @property
    def palm_anchors(self):
        finger_joints = self.joints[..., self.finger_indices_in_hand, :] # (B, T, 5, 4, 3)
        wrist_joints = self.joints[..., [0], :] # (B, T, 1, 3)
        anchors = torch.cat([
            finger_joints[..., :-1, 0, :] * 1/3 + wrist_joints * 2/3, # (B, T, 4, 3)
            finger_joints[..., :-1, 0, :] * 2/3 + wrist_joints * 1/3, # (B, T, 4, 3)
            finger_joints[..., [-1], 0, :] * 1/2 + wrist_joints * 1/2, # (B, T, 1, 3)
        ], dim=-2)
        return anchors # (B, T, 9, 3)


    @property
    def hand_rotations(self):
        '''
        return: (B, T, J_hand, 4)
        '''
        assert self._hand_rotations is not None
        return self._hand_rotations


    @hand_rotations.setter
    def hand_rotations(self, value: torch.Tensor):
        '''
        value: (B, T, J_hand, 4)
        '''
        assert self.rest_pose is not None and self.hand_offsets is not None and self.m_tbs is not None

        root_rotation = torch.zeros_like(value[..., [0], :]) # Set the global rotation to identity
        root_rotation[..., 0] = 1.0
        self._hand_rotations = torch.cat([root_rotation, value[..., -20:, :].clone()], dim=-2) # (B, T, J_hand, 4)

        B, T = self._hand_rotations.shape[:2]
        hand_rotations_global = self.fk.forward(self._hand_rotations, torch.zeros(B, T, 3, dtype=self._hand_rotations.dtype, device=self._hand_rotations.device), self.hand_offsets, ret_rot=True, ret_ori_repr=True)[0]

        finger_rotations = self._hand_rotations[..., self.finger_indices_in_hand[:, :3], :] # Ignore finger tip (B, T, 5, 3, 4)
        for finger_idx, finger_j_indices in enumerate(self.finger_indices_in_hand[:4]): # Finger root joint inherits the global rotation, except for the thumb
            finger_rotations[..., finger_idx, 0, :] = hand_rotations_global[..., finger_j_indices[0], :]
        finger_rest_pose = self.rest_pose[..., self.finger_indices_in_hand[:, :3], :].to(value.device, value.dtype) # Ignore finger tip (B, 1, 5, 3, 4)
        tbs_rotations = matrix_to_quaternion(self.m_tbs.to(self._hand_rotations.device, self._hand_rotations.dtype)[..., :3, :, :]).unsqueeze(1) # (B, 1, 5, 3, 4)
        rest_tbs_rotations = quaternion_multiply(finger_rest_pose, tbs_rotations)
        finger_rotations = quaternion_multiply(finger_rotations, quaternion_invert(finger_rest_pose))
        finger_rotations = quaternion_multiply(finger_rotations, rest_tbs_rotations)
        finger_rotations = quaternion_multiply(quaternion_invert(rest_tbs_rotations), finger_rotations)
        self._bs_rotations = standardize_quaternion(finger_rotations)


    @property
    def bs_rotations(self):
        assert self._bs_rotations is not None
        return self._bs_rotations


    @bs_rotations.setter
    def bs_rotations(self, value: torch.Tensor):
        '''
        bs_rotations: (B, T, N_finger, 3, 4)
        '''
        assert self.rest_pose is not None and self.hand_offsets is not None and self.m_tbs is not None
        self._bs_rotations = value.clone()

        B, T = value.shape[:2]
        J_hand = len(self.hand_j_ori_indices)

        hand_rotations = torch.zeros(B, T, J_hand, 4, dtype=value.dtype, device=value.device) # Initialize hand_rotations with identity
        hand_rotations[..., 0] = 1.0

        finger_rotations = value.clone()
        finger_rest_pose = self.rest_pose[..., self.finger_indices_in_hand[:, :3], :].to(value.device, value.dtype)
        tbs_rotations = matrix_to_quaternion(self.m_tbs.to(value.device, value.dtype)[..., :3, :, :]).unsqueeze(1) # (B, 1, 5, 3, 4)
        rest_tbs_rotations = quaternion_multiply(finger_rest_pose, tbs_rotations)
        finger_rotations = quaternion_multiply(finger_rotations, quaternion_invert(rest_tbs_rotations))
        finger_rotations = quaternion_multiply(rest_tbs_rotations, finger_rotations)
        finger_rotations = quaternion_multiply(finger_rotations, finger_rest_pose)
        hand_rotations[..., self.finger_indices_in_hand[:, :3], :] = finger_rotations

        self._hand_rotations = standardize_quaternion(hand_rotations)


    @property
    def joints(self):
        raise NotImplementedError()


    def copy_hand_data(self, rotations: torch.Tensor, hand_rotations: torch.Tensor):
        '''
        rotations: (B, T, J_body, 4)
        hand_rotations: (B, T, J_hand, 4)
        return: (B, T, J_body, 4)
        '''
        new_rotations = rotations.clone()
        new_rotations[..., self.hand_j_ori_indices[1:], :] = hand_rotations[..., 1:, :] # Ignore wrist
        return new_rotations


class BEATArmature(BaseArmature):
    def __init__(self, is_rhand: bool, speaker_ids: Optional[List[int]] = None):
        if speaker_ids is not None:
            with open('artifact/beat/beat4_v0.2.1/coordiantes.pkl', 'rb') as f:
                finger_axes = pickle.load(f)
            hand_key = 'right' if is_rhand else 'left'
            finger_axes = [torch.from_numpy(finger_axes[i][hand_key]) for i in speaker_ids]
            finger_axes = torch.stack(finger_axes, dim=0)
        else:
            finger_axes = None
        super().__init__(self.beat_joint_names, self.beat_parents, self.beat_hand_labels, finger_axes, is_rhand)

    beat_hand_labels = [
        [
            'RightHand', # Wrist
            ['RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4'], # Middle
            ['RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4'], # Ring
            ['RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4'], # Pinky
            ['RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4'], # Index
            ['RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4'], # Thumb
            ['RightHandRing', 'RightHandPinky', 'RightHandIndex'] # Palm
        ],
        [
            'LeftHand', # Wrist
            ['LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4'], # Middle
            ['LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4'], # Ring
            ['LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4'], # Pinky
            ['LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4'], # Index
            ['LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4'], # Thumb
            ['LeftHandRing', 'LeftHandPinky', 'LeftHandIndex'] # Palm
        ]
    ]

    beat_joint_names = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'HeadEnd', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightForeFoot', 'RightToeBase', 'RightToeBaseEnd', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftForeFoot', 'LeftToeBase', 'LeftToeBaseEnd']

    beat_parents = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 13, 14, 15, 12, 17, 18, 19, 20, 17, 22, 23, 24, 25, 12, 27, 28, 29, 30, 27, 32, 33, 34, 4, 36, 37, 38, 39, 40, 41, 42, 39, 44, 45, 46, 47, 44, 49, 50, 51, 52, 39, 54, 55, 56, 57, 54, 59, 60, 61, 0, 63, 64, 65, 66, 67, 0, 69, 70, 71, 72, 73]


class MixamoArmature(BaseArmature):
    def __init__(self, is_rhand: bool, finger_axes: Union[Dict[str, np.ndarray], torch.Tensor, None] = None, mesh_data: Optional[Dict[str, np.ndarray]] = None, hand_offsets: Optional[torch.Tensor] = None):
        if finger_axes is None:
            finger_axes = torch.zeros(1, 5, 4, 3, 3)
        elif isinstance(finger_axes, dict):
            hand_key = 'rhand' if is_rhand else 'lhand'
            finger_axes = torch.from_numpy(finger_axes[hand_key])
            if len(finger_axes.shape) == 3:
                finger_axes = finger_axes.unsqueeze(1).repeat(1, 4, 1, 1) # (5, 4, 3, 3)
            finger_axes = finger_axes.unsqueeze(0) # (1, 5, 4, 3, 3)
        elif isinstance(finger_axes, torch.Tensor): # given batch of finger axes
            if len(finger_axes.shape) == 3:
                finger_axes = finger_axes.unsqueeze(2).repeat(1, 4, 1, 1) # (B, 5, 4, 3, 3)
        else:
            raise ValueError(f'finger_axes should be None, dict or torch.Tensor, but got {type(finger_axes)}')

        super().__init__(self.mixamo_joint_names, self.mixamo_parents, self.mixamo_hand_labels, finger_axes, is_rhand)

        if mesh_data is not None and hand_offsets is not None:
            raise ValueError('mesh_data and hand_offsets cannot be both given')

        if mesh_data is not None:
            hand_mesh, lbs_weight = extract_joint_mesh(mesh_data, self.hand_j_labels, mask_threshold=0.2, return_weight=True)
            self.hand_verts = np.asarray(hand_mesh.vertices)
            self.hand_faces = np.asarray(hand_mesh.triangles)
            self.hand_lbs_weight = lbs_weight
            vgrp_cors = mesh_data['vgrp_cors']
            vgrp_names = mesh_data['vgrp_label'].tolist()
            hand_j_in_vgrp = [vgrp_names.index(j) for j in self.hand_j_labels]
            parent_names = [self.hand_j_labels[self.hand_parents[i]] for i in range(len(self.hand_j_labels))]
            parent_j_in_vgrp = [vgrp_names.index(j) for j in parent_names]
            vgrp_offsets = vgrp_cors[hand_j_in_vgrp] - vgrp_cors[parent_j_in_vgrp]
            vgrp_offsets[0] = np.zeros(3)
            hand_offsets = torch.from_numpy(vgrp_offsets).unsqueeze(0) # (B, J_hand, 3)
            self.build_finger_frame(hand_offsets)
            self.verts_origin = vgrp_cors[hand_j_in_vgrp[0]]
            self.hand_joint_cors = vgrp_cors[hand_j_in_vgrp]
        elif hand_offsets is not None: # given hand_offsets in batch
            self.build_finger_frame(hand_offsets)
        else:
            raise ValueError('mesh_data and hand_offsets cannot be both None')

    @property
    def verts(self):
        assert self.m_tbs.shape[0] == 1 # assert batch_size is 1
        skin_mesh = SkinnableMesh(self.hand_verts, self.hand_joint_cors, self.hand_parents, self.hand_lbs_weight)
        hand_pose = self._hand_rotations
        posed_verts = skin_mesh.skin(hand_pose)
        verts_origin = torch.from_numpy(self.verts_origin).unsqueeze(0).unsqueeze(0).to(posed_verts.device, posed_verts.dtype)
        return posed_verts - verts_origin

    @property
    def joints(self):
        B, T = self._hand_rotations.shape[:2]
        root_positions = torch.zeros(B, T, 3, device=self._hand_rotations.device, dtype=self._hand_rotations.dtype)
        return self.fk.forward(self._hand_rotations, root_positions, self.hand_offsets.to(self._hand_rotations.device))

    @property
    def faces(self):
        return self.hand_faces

    mixamo_hand_labels = None

    mixamo_joint_names = None

    mixamo_parents = None


class MANOArmature(BaseArmature):
    def __init__(self, is_rhand: bool, smplx_path: str, finger_axes: Union[Dict[str, np.ndarray], None] = None, shape: Optional[torch.Tensor] = None):
        self.is_rhand = is_rhand
        self.smplx_path = smplx_path
        self.finger_axes_data = finger_axes
        self.shape = shape

        if is_rhand:
            joint_names = [f'right_{n}' for n in self.mano_joint_names]
        else:
            joint_names = [f'left_{n}' for n in self.mano_joint_names]

        batch_size = 1 if shape is None else shape.shape[0]
        mano_layer: Dict[str, smplx.MANO] = {
            'right': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True, batch_size=batch_size),
            'left': smplx.create(self.smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True, batch_size=batch_size)
            }
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
            # print('Fix shapedirs bug of MANO')
            mano_layer['left'].shapedirs[:,0,:] *= -1
        self.mano_model = mano_layer['right'] if is_rhand else mano_layer['left']
        self.mano_model.requires_grad_(False)
        self.mano_model.eval()
        self.J_regressor_ext = torch.zeros(len(self.mano_joint_names), self.mano_model.J_regressor.shape[1], device=self.mano_model.J_regressor.device, dtype=self.mano_model.J_regressor.dtype)
        self.J_regressor_ext[:-5] = self.mano_model.J_regressor
        for i, v in enumerate(self.mano_vertex_ids):
            self.J_regressor_ext[-5+i, v] = 1

        super().__init__(joint_names, self.mano_parents, self.mano_hand_labels, None, is_rhand)

        finger_axes = self.calculate_finger_axes()

        super().__init__(joint_names, self.mano_parents, self.mano_hand_labels, finger_axes, is_rhand)

        self.hand_j_ori_in_arm = [self.hand_j_labels.index(l) for l in self.joint_names]

        self.build_finger_frame(self.offsets)

    def calculate_finger_axes(self):
        batch_size = self.shape.shape[0] if self.shape is not None else 1
        if isinstance(self.finger_axes_data, dict) and self.shape is not None:
            hand_key = 'rhand' if self.is_rhand else 'lhand'
            if f'{hand_key}_bend_bary' in self.finger_axes_data:
                hand_pose = torch.zeros(batch_size, 45, device=self.shape.device, dtype=self.shape.dtype)
                global_orient = torch.zeros(batch_size, 3, device=self.shape.device, dtype=self.shape.dtype)
                verts = self.mano_model(hand_pose=hand_pose, global_orient=global_orient, betas=self.shape, return_verts=True).vertices.detach().cpu().numpy() # potential bug, here shape does not affect the finger axes
                bend_axes = []
                for bary, index_tri in zip(self.finger_axes_data[f'{hand_key}_bend_bary'], self.finger_axes_data[f'{hand_key}_bend_index_tri']):
                    bend_axes.append(bary_to_axis_batch(verts.copy(), self.mano_model.faces.copy(), bary, index_tri))
                bend_axes = np.stack(bend_axes, axis=1) # [batch_size, 5, 3]
                bend_axes = np.tile(bend_axes[..., np.newaxis, :], [1, 1, 4, 1]) # [batch_size, 5, 4, 3]
                twist_axes = []
                for finger_indices in self.finger_indices_in_hand:
                    cur_finger_twist_axes = self.offsets[..., finger_indices[1:], :]
                    cur_finger_twist_axes = torch.cat([cur_finger_twist_axes, cur_finger_twist_axes[..., -1:, :]], dim=-2)
                    twist_axes.append(cur_finger_twist_axes.cpu().numpy())
                twist_axes = np.stack(twist_axes, axis=1) # [batch_size, 5, 4, 3]
                twist_axes = twist_axes / np.linalg.norm(twist_axes, axis=-1, keepdims=True)
                bend_axes = bend_axes - np.sum(bend_axes * twist_axes, axis=-1, keepdims=True) * twist_axes
                bend_axes = bend_axes / np.linalg.norm(bend_axes, axis=-1, keepdims=True)
                splay_axes = np.cross(twist_axes, bend_axes)
                mask = splay_axes[..., 1] < 0
                splay_axes[mask] *= -1
                bend_axes = np.cross(splay_axes, twist_axes)
                finger_axes = np.stack([twist_axes, bend_axes, splay_axes], axis=-2) # [batch_size, 5, 4, 3, 3]
                finger_axes = torch.from_numpy(finger_axes).to(self.shape.device) # [batch_size, 5, 4, 3, 3]
            else:
                finger_axes = torch.from_numpy(self.finger_axes_data[hand_key]).unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, 4, 1, 1)
        else:
            finger_axes = torch.zeros(1, 5, 4, 3, 3)

        return finger_axes


    def reset_shape(self, shape: torch.Tensor):
        self.shape = shape
        self.mano_model = self.mano_model.to(shape.device)
        finger_axes = self.calculate_finger_axes()
        self.m_tbs = finger_axes.transpose(-1, -2)
        self.build_finger_frame(self.offsets)

    @property
    def joints(self) -> torch.Tensor:
        if self._hand_rotations is None:
            batch_size = self.shape.shape[0] if self.shape is not None else 1
            hand_pose = torch.zeros(batch_size, 45, device=self.mano_model.J_regressor.device, dtype=self.mano_model.J_regressor.dtype)
            global_orient = torch.zeros(batch_size, 3, device=self.mano_model.J_regressor.device, dtype=self.mano_model.J_regressor.dtype)
            output = self.mano_model(hand_pose=hand_pose, global_orient=global_orient, betas=self.shape, return_verts=True)
            ori_j_cors = torch.einsum('ji,bik->bjk', self.J_regressor_ext.to(output.vertices.device), output.vertices)
        else:
            hand_rotations = self._hand_rotations[..., self.hand_j_ori_in_arm, :]
            pose = quaternion_to_axis_angle(hand_rotations)
            B, T = pose.shape[:2]
            global_orient = pose[..., 0, :]
            hand_pose = pose[..., 1:-5, :].reshape(pose.shape[:-2] + (5, 3, 3)) # (B, T, 5, 3, 3) remove finger tip
            hand_pose = hand_pose.reshape(pose.shape[:-2] + (-1,)) # (B, T, 45)
            betas = self.shape.unsqueeze(1).repeat(1, T, 1) # (B, T, 10)
            global_orient = global_orient.reshape(-1, 3) # (B*T, 3)
            hand_pose = hand_pose.reshape(-1, 45) # (B*T, 45)
            betas = betas.reshape(-1, 10) # (B*T, 10)
            output = self.mano_model(betas=betas, global_orient=global_orient, hand_pose=hand_pose, return_verts=True)
            vertices = output.vertices.reshape(B, T, -1, 3) # (B, T, V, 3)
            ori_j_cors = torch.einsum('ji,btik->btjk', self.J_regressor_ext.to(vertices.device), vertices)

        return ori_j_cors[..., self.hand_j_ori_indices, :] - ori_j_cors[..., [0], :]
    
    @property
    def control_points(self) -> torch.Tensor:
        '''
        return: [batch_size, 5, 3, 3]
        '''
        joints = self.joints[..., self.finger_indices_in_hand, :] # (B, T, 5, 4, 3)
        control_pts = (joints[..., :-1] + joints[..., 1:]) / 2 # (B, T, 5, 3, 3)
        return control_pts


    @property
    def verts(self) -> torch.Tensor:
        if self._hand_rotations is None:
            output = self.mano_model(betas=self.shape, return_verts=True)
            vertices = output.vertices.unsqueeze(1) # (B, T, V, 3)
        else:
            hand_rotations = self._hand_rotations[..., self.hand_j_ori_in_arm, :]
            pose = quaternion_to_axis_angle(hand_rotations)
            B, T = pose.shape[:2]
            global_orient = pose[..., 0, :] # (B, T, 3)
            hand_pose = pose[..., 1:-5, :].reshape(pose.shape[:-2] + (5, 3, 3)) # (B, T, 5, 3, 3) remove finger tip
            hand_pose = hand_pose.reshape(pose.shape[:-2] + (-1,)) # (B, T, 45)
            betas = self.shape.unsqueeze(1).repeat(1, T, 1) # (B, T, 10)
            global_orient = global_orient.reshape(-1, 3) # (B*T, 3)
            hand_pose = hand_pose.reshape(-1, 45) # (B*T, 45)
            betas = betas.reshape(-1, 10) # (B*T, 10)
            output = self.mano_model(betas=betas, global_orient=global_orient, hand_pose=hand_pose, return_verts=True)
            vertices = output.vertices.reshape(B, T, -1, 3)

        wrist_cors = torch.einsum('i,btik->btk', self.J_regressor_ext[0].to(vertices.device), vertices).unsqueeze(2)
        return (vertices - wrist_cors)

    @property
    def faces(self):
        return self.mano_model.faces

    @property
    def offsets(self) -> torch.Tensor:
        ori_hand_rotations = self._hand_rotations
        self._hand_rotations = None # Make sure the hand is in rest pose
        joints = self.joints
        self._hand_rotations = ori_hand_rotations
        offsets = joints[..., 1:, :] - joints[..., self.hand_parents[1:], :]
        offsets = torch.cat([torch.zeros_like(offsets[..., :1, :]), offsets], dim=-2)
        return offsets

    def check_self_intersection(self, return_names: bool = False, return_data: bool = False):
        assert self.shape.shape[0] == 1, 'Only support batch size 1'

        finger_labels = [
            ['index1', 'index2', 'index3'],
            ['middle1', 'middle2', 'middle3'],
            ['ring1', 'ring2', 'ring3'],
            ['pinky1', 'pinky2', 'pinky3'],
            ['thumb1', 'thumb2', 'thumb3']
        ]

        def extract_mano_joint_mesh(mano_model: smplx.MANO, vertices: torch.Tensor, joint_names: list):
            mesh_data = {
                'verts': vertices.detach().cpu().numpy().squeeze(),
                'faces': mano_model.faces,
                'weight': mano_model.lbs_weights.detach().cpu().numpy().squeeze()
            }

            wrist_label = 'right_wrist' if mano_model.is_rhand else 'left_wrist'
            finger_label_start_idx = SMPLH_JOINT_NAMES.index('right_index1') if mano_model.is_rhand else SMPLH_JOINT_NAMES.index('left_index1')
            finger_label = SMPLH_JOINT_NAMES[finger_label_start_idx:finger_label_start_idx+15]

            mesh_data['vgrp_label'] = np.array([wrist_label] + finger_label)

            joint_mesh = extract_joint_mesh(mesh_data, [f'right_{j}' for j in joint_names] if mano_model.is_rhand else [f'left_{j}' for j in joint_names], 0.4)
            return joint_mesh

        meshes = [extract_mano_joint_mesh(self.mano_model, self.verts, labels) for labels in finger_labels]

        collision_manager = CollisionManager()
        for mesh, labels in zip(meshes, finger_labels):
            collision_manager.add_object(labels[0][:-1], Trimesh(mesh.vertices, mesh.triangles))
        return collision_manager.in_collision_internal(return_names, return_data)


    mano_hand_labels = [
        [
            'right_wrist',
            ['right_middle1', 'right_middle2', 'right_middle3', 'right_middle4'],
            ['right_ring1', 'right_ring2', 'right_ring3', 'right_ring4'],
            ['right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky4'],
            ['right_index1', 'right_index2', 'right_index3', 'right_index4'],
            ['right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb4']
        ],
        [
            'left_wrist',
            ['left_middle1', 'left_middle2', 'left_middle3', 'left_middle4'],
            ['left_ring1', 'left_ring2', 'left_ring3', 'left_ring4'],
            ['left_pinky1', 'left_pinky2', 'left_pinky3', 'left_pinky4'],
            ['left_index1', 'left_index2', 'left_index3', 'left_index4'],
            ['left_thumb1', 'left_thumb2', 'left_thumb3', 'left_thumb4']
        ]
    ]

    mano_joint_names = ['wrist', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3', 'index4', 'middle4', 'pinky4', 'ring4', 'thumb4']

    mano_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]

    mano_vertex_ids = [333, 444, 672, 555, 744] # vertices id of tip joints


class AnotomicalAxisCalculator:
    def __init__(self, finger_j_indices: List[List[int]], hand_j_offsets: np.ndarray):
        self.finger_j_indices = finger_j_indices
        self.finger_bend_axes_data = [[[], []] if i != 4 else [[]] for i in range(len(finger_j_indices))]
        self.hand_j_offsets = hand_j_offsets


    def add_rotation_data(self, hand_rotations: np.ndarray):
        for finger_idx, cur_finger_j_indices in enumerate(self.finger_j_indices):
            if finger_idx != 4:
                interested_j_indices = cur_finger_j_indices[1:3] # Ignore finger tip and metacarpophalangeal joint
            else:
                interested_j_indices = cur_finger_j_indices[2:3] # Thumb has only 1 interested joint
            for i, j_idx in enumerate(interested_j_indices):
                j_bend_axes = hand_rotations[:, j_idx, 1:]
                j_bend_axes = j_bend_axes[np.linalg.norm(j_bend_axes, axis=-1) > 1e-4]
                j_bend_axes = j_bend_axes / np.linalg.norm(j_bend_axes, axis=-1, keepdims=True)
                mask = (j_bend_axes[:, 2] < 0)
                j_bend_axes[mask] = -j_bend_axes[mask] # Make sure the bend axis is pointing positive z-axis
                self.finger_bend_axes_data[finger_idx][i].append(j_bend_axes)


    @property
    def finger_axis(self):
        if len(self.finger_bend_axes_data[0][0]) == 0:
            return None

        finger_bend_axes = []
        for finger_i, b in enumerate(self.finger_bend_axes_data):
            cur_finger_bend_axes = []
            for j_i, axes in enumerate(b):
                axes = np.concatenate(axes, axis=0)
                mean_bend_axis = np.mean(axes, axis=0)
                mean_bend_axis = mean_bend_axis / np.linalg.norm(mean_bend_axis)
                cur_finger_bend_axes.append(mean_bend_axis)
                std = np.std(axes, axis=0)
                print(f'Finger {finger_i} Joint {j_i} std: {std}')
                if std.max() > 1e-4:
                    raise ValueError(f'Finger {finger_i} Joint {j_i} bend axis is not stable')
            while len(cur_finger_bend_axes) < 3: # Add bend axis for the metacarpophalangeal joint
                cur_finger_bend_axes.insert(0, cur_finger_bend_axes[0].copy())
            cur_finger_bend_axes.append(cur_finger_bend_axes[-1].copy()) # Add bend axis for the finger tip
            finger_bend_axes.append(cur_finger_bend_axes)

        finger_bend_axes = np.stack(finger_bend_axes, axis=0) # (5, 4, 3)
        finger_j_offsets = self.hand_j_offsets[self.finger_j_indices] # (5, 4, 3)
        finger_twist_axes = np.concatenate([finger_j_offsets[:, 1:], finger_j_offsets[:, -1:]], axis=1) # (5, 4, 3)
        finger_twist_axes = finger_twist_axes / np.linalg.norm(finger_twist_axes, axis=-1, keepdims=True)

        for finger_j in range(finger_bend_axes.shape[0]):
            if np.sum(finger_bend_axes[finger_j] * finger_twist_axes[finger_j], axis=-1).max() > 1e-4:
                raise ValueError(f'Finger {finger_j} twist axis and bend axis are not perpendicular')

        finger_bend_axes -= np.sum(finger_bend_axes * finger_twist_axes, axis=-1, keepdims=True) * finger_twist_axes # Make sure the bend axis is perpendicular to the twist axis
        finger_bend_axes = finger_bend_axes / np.linalg.norm(finger_bend_axes, axis=-1, keepdims=True)

        finger_splay_axes = np.cross(finger_twist_axes, finger_bend_axes, axis=-1) # (5, 4, 3)

        mask = (finger_splay_axes[:, :, 1] < 0)
        finger_splay_axes[mask] = -finger_splay_axes[mask] # Make sure the splay axis is pointing to the top
        finger_bend_axes[mask] = -finger_bend_axes[mask] # Make sure the splay axis is pointing to the top

        finger_axes = np.stack([finger_twist_axes, finger_bend_axes, finger_splay_axes], axis=-2) # (5, 4, 3, 3)

        return finger_axes


def display_armature(armature: BaseArmature):
    tbs_matrix = armature.tbs_matrix_global()
    joints = armature.joints.detach()[0, 0].cpu().numpy().squeeze()
    factor = 1 / np.linalg.norm(joints[0] - joints[1])
    mesh = Trimesh(vertices=armature.verts.detach()[0, 0].cpu().numpy().squeeze() * factor, faces=armature.faces)
    joints = joints * factor
    paths = []
    for i in range(20):
        x_path = load_path_from_axis(tbs_matrix[..., i//4, i%4, :, 0].detach()[0, 0].cpu().numpy().squeeze(), joints[i+1], color=np.array([255, 0, 0, 255]), n_points=3)
        y_path = load_path_from_axis(tbs_matrix[..., i//4, i%4, :, 1].detach()[0, 0].cpu().numpy().squeeze(), joints[i+1], color=np.array([0, 255, 0, 255]), n_points=3)
        z_path = load_path_from_axis(tbs_matrix[..., i//4, i%4, :, 2].detach()[0, 0].cpu().numpy().squeeze(), joints[i+1], color=np.array([0, 0, 255, 255]), n_points=3)
        paths.extend([x_path, y_path, z_path])
    trimesh.Scene([mesh] + paths).show()


def export_armature(armature: BaseArmature, p: str, factor: float = 1.0, f: int = 0):
    if armature.verts.shape[0] == 1:
        verts = armature.verts.detach()[0, f].cpu().numpy().squeeze()
    else:
        verts = armature.verts.detach()[f].cpu().numpy().squeeze()
    mesh = Trimesh(vertices=verts * factor, faces=armature.faces)
    with open(p, 'w') as f:
        f.write(export_obj(mesh))


def export_armature_animation(armature: BaseArmature, p: str, show_anchor: bool = False, animation: bool = True, color: str = 'white'):
    verts = armature.verts.detach().cpu().numpy().squeeze()
    anchors = armature.palm_anchors.detach().cpu().numpy().squeeze()
    joints = armature.joints.detach().cpu().numpy().squeeze()
    mesh = pv.wrap(Trimesh(vertices=verts[0], faces=armature.faces))
    if animation:
        plotter = pv.Plotter(off_screen=True)
    else:
        plotter = pv.Plotter()
    plotter.camera.roll = 0
    if isinstance(armature, MixamoArmature):
        plotter.camera.position = (-60.0, 0.0, 0.0)
        verts[..., 0] += 10.0
        joints[..., 0] += 10.0
        anchors[..., 0] += 10.0
        plotter.camera.elevation = -90.0
    elif isinstance(armature, MANOArmature):
        plotter.camera.position = (-0.5, 0.0, 0.0)
        verts[..., 0] += 0.08
        joints[..., 0] += 0.08
        anchors[..., 0] += 0.08
        plotter.camera.elevation = -90.0
    if show_anchor:
        plotter.add_mesh(mesh, color=color, smooth_shading=True, opacity=0.4)
    else:
        plotter.add_mesh(mesh, color=color, smooth_shading=False, opacity=1.0)
    plotter.set_background('white')
    plotter.add_camera_orientation_widget()

    anchor_size = 0.5 if isinstance(armature, MixamoArmature) else 0.005
    if show_anchor:
        anchor_actors = []
        for i in range(9):
            a = plotter.add_mesh(pv.Cube(center=anchors[0, i], x_length=anchor_size, y_length=anchor_size, z_length=anchor_size), color='yellow')
            anchor_actors.append(a)

    if animation:
        plotter.image_scale = 4
        plotter.open_movie(p, framerate=5)

        for v_idx, v in enumerate(verts):
            plotter.update_coordinates(v, mesh)

            if show_anchor:
                for a in anchor_actors:
                    plotter.remove_actor(a)
                anchor_meshes = [pv.Cube(center=anchors[v_idx, i], x_length=anchor_size, y_length=anchor_size, z_length=anchor_size) for i in range(9)]
                for m in anchor_meshes:
                    a = plotter.add_mesh(m, color='yellow')
                    anchor_actors.append(a)

            plotter.write_frame()

        plotter.close()

    else:
        plotter.show()


def show_armature(armature: BaseArmature, frame: int = 0, color: str = 'white', roll: int = 0):
    verts = armature.verts.detach().cpu().numpy().squeeze()
    anchors = armature.palm_anchors.detach().cpu().numpy().squeeze()
    joints = armature.joints.detach().cpu().numpy().squeeze()
    plotter = pv.Plotter()
    plotter.camera.roll = roll
    if isinstance(armature, MixamoArmature):
        plotter.camera.position = (-60.0, 0.0, 0.0)
        verts[..., 0] += 10.0
        joints[..., 0] += 10.0
        anchors[..., 0] += 10.0
        plotter.camera.elevation = -90.0
    elif isinstance(armature, MANOArmature):
        plotter.camera.position = (-0.5, 0.0, 0.0)
        verts[..., 0] += 0.08
        joints[..., 0] += 0.08
        anchors[..., 0] += 0.08
        plotter.camera.elevation = -90.0
    mesh = pv.wrap(Trimesh(vertices=verts[frame], faces=armature.faces))

    plotter.add_mesh(mesh, color=color, smooth_shading=False, opacity=1.0)
    plotter.set_background('white')
    plotter.add_camera_orientation_widget()
    plotter.image_scale = 4
    plotter.show()
    return plotter.screenshot(transparent_background=True, return_img=True)


def export_bvh(armature: BaseArmature, p: str, ref_bvh: str, start_frame: int = 0):
    ref_anim, ref_joint_names, ref_frame_time = BVH.load(ref_bvh)
    joint_names = armature.hand_j_labels[1:] # do not copy wrist rotation
    rotations = armature.hand_rotations.cpu().numpy().squeeze()

    for j_idx, j_name in enumerate(joint_names):
        if j_name in ref_joint_names:
            ref_idx = ref_joint_names.index(j_name)
            T = rotations.shape[0]
            ref_anim.rotations.qs[start_frame:start_frame+T, ref_idx] = rotations[:, j_idx]

    BVH.save(p, ref_anim, ref_joint_names, ref_frame_time)
