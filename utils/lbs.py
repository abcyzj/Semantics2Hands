from typing import Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.transforms import (euler_angles_to_matrix, matrix_to_quaternion,
                                  quaternion_to_matrix)
from trimesh import Trimesh

from .mesh import extract_joint_mesh


def rigid_transform(
    rot_mat: torch.Tensor, # (N, J, 3, 3)
    joint_locations: torch.Tensor, # (N, J, 3)
    parents: Union[torch.Tensor, np.ndarray] # (J)
):
    '''
    rot_mat: (N, J, 3, 3)
    joint_locations: (N, J, 3)
    parents: (J)

    Returns
    posed_joint_global: (N, J, 3)
    pose_transform_in_g: (N, J, 4, 4)
    '''
    j_in_p = joint_locations.clone().unsqueeze(-1) # (N, J, 3, 1) joint in parent frame
    j_in_p = torch.cat([j_in_p[:, [0]], j_in_p[:, 1:] - j_in_p[:, parents[1:]]], dim=1)

    jf_in_p = torch.cat([F.pad(rot_mat, [0, 0, 0, 1]), F.pad(j_in_p, [0, 0, 0, 1], value=1.0)], dim=3) # (N, J, 4, 4) joint frame in parent frame

    jf_in_g = [jf_in_p[:, 0]]
    for j_idx, p_idx in enumerate(parents):
        if p_idx == -1:
            assert j_idx == 0
            continue
        jf_in_g.append(jf_in_g[p_idx] @ jf_in_p[:, j_idx])
    jf_in_g = torch.stack(jf_in_g, dim=1) # (N, J, 4, 4) joint frame in global frame

    posed_joint_global = jf_in_g[:, :, :3, 3]

    N, J = joint_locations.shape[:2]
    gf_in_rest_j = torch.eye(4, 3, dtype=joint_locations.dtype, device=joint_locations.device).repeat(N, J, 1, 1) # (N, J, 4, 3)
    gf_in_rest_j = torch.cat([gf_in_rest_j, F.pad(-joint_locations.clone().unsqueeze(-1), [0, 0, 0, 1], value=1)], dim=3) # (N, J, 4, 4) global frame in rest pose joint frame, the orientation of rest joint frame is the same as the global frame

    pose_transform_in_g = jf_in_g @ gf_in_rest_j # (N, J, 4, 4) transform in global frame

    return posed_joint_global, pose_transform_in_g



def lbs(
    pose_mat: Union[torch.Tensor, np.ndarray],  # (*, J, 3, 3)
    joint_locations: Union[torch.Tensor, np.ndarray],  # (*, J, 3)
    parents: Union[torch.Tensor, np.ndarray],  # (J)
    verts: Union[torch.Tensor, np.ndarray],  # (*, V, 3)
    lbs_weights: Union[torch.Tensor, np.ndarray],  # (*, V, J)
    device: torch.device = torch.device('cpu')
):
    '''
    Return
    posed_joint_global: (*, J, 3)
    posed_v: (*, V, 3)
    '''
    if isinstance(pose_mat, np.ndarray):
        pose_mat = torch.from_numpy(pose_mat).to(device)
    if isinstance(joint_locations, np.ndarray):
        joint_locations = torch.from_numpy(joint_locations).to(device)
    if isinstance(verts, np.ndarray):
        verts = torch.from_numpy(verts).to(device)
    if isinstance(lbs_weights, np.ndarray):
        lbs_weights = torch.from_numpy(lbs_weights).to(device)

    ori_shape = pose_mat.shape[:-3]
    J = joint_locations.shape[-2]
    V = verts.shape[-2]
    pose_mat = pose_mat.reshape(-1, J, 3, 3)
    joint_locations = joint_locations.reshape(-1, J, 3)
    verts = verts.reshape(-1, V, 3)
    lbs_weights = lbs_weights.reshape(-1, V, J)

    posed_joint_global, pose_transform_in_g = rigid_transform(pose_mat, joint_locations, parents)

    weighted_transform_in_g = torch.einsum('nvj,njmk->nvmk', lbs_weights, pose_transform_in_g) # (N, V, 4, 4)
    v_homo = F.pad(verts, [0, 1], value=1.0).unsqueeze(-1) # (N, V, 4, 1)
    posed_v_homo = weighted_transform_in_g @ v_homo # (N, V, 4, 1)
    posed_v = posed_v_homo.squeeze(-1)[:, :, :3] # (N, V, 3)

    return posed_joint_global.reshape(ori_shape + (J, 3)), posed_v.reshape(ori_shape + (V, 3))


class SkinnableMesh:
    def __init__(self, verts: np.ndarray, joint_cors: np.ndarray, parents: np.ndarray, weight: np.ndarray):
        self.verts = verts.copy()
        self.joint_locations = joint_cors.copy()
        self.parents = parents.copy()
        self.lbs_weights = weight.copy()

    def skin(self, pose: Union[torch.Tensor, np.ndarray]):
        '''
        pose: (*, J, 3, 3) or (*, J, 4)
        '''
        ret_numpy = False
        if isinstance(pose, np.ndarray):
            ret_numpy = True
            pose = torch.from_numpy(pose)

        if pose.shape[-1] == 4:
            pose = quaternion_to_matrix(pose)

        ori_shape = pose.shape[:-3]
        pose = pose.reshape((-1,) + pose.shape[-3:])

        joint_locations = torch.from_numpy(self.joint_locations).repeat(ori_shape + (1, 1)).to(pose.device, pose.dtype)
        parents = torch.from_numpy(self.parents).to(pose.device)
        verts = torch.from_numpy(self.verts).repeat(ori_shape + (1, 1)).to(pose.device, pose.dtype)
        lbs_weights = torch.from_numpy(self.lbs_weights).repeat(ori_shape + (1, 1)).to(pose.device, pose.dtype)
        _, posed_verts = lbs(pose, joint_locations, parents, verts, lbs_weights)

        posed_verts = posed_verts.reshape(ori_shape + posed_verts.shape[-2:])
        if ret_numpy:
            posed_verts = posed_verts.cpu().numpy()

        return posed_verts


if __name__ == '__main__':
    mesh_data = np.load('artifact/remy_mesh_data.npz')
    hand_mesh = extract_joint_mesh(mesh_data, [
            'mixamorig:RightHand', # Wrist
            'mixamorig:RightHandMiddle1', 'mixamorig:RightHandMiddle2', 'mixamorig:RightHandMiddle3', 'mixamorig:RightHandMiddle4', # Middle
            'mixamorig:RightHandRing1', 'mixamorig:RightHandRing2', 'mixamorig:RightHandRing3', 'mixamorig:RightHandRing4', # Ring
            'mixamorig:RightHandPinky1', 'mixamorig:RightHandPinky2', 'mixamorig:RightHandPinky3', 'mixamorig:RightHandPinky4', # Pinky
            'mixamorig:RightHandIndex1', 'mixamorig:RightHandIndex2', 'mixamorig:RightHandIndex3', 'mixamorig:RightHandIndex4', # Index
            'mixamorig:RightHandThumb1', 'mixamorig:RightHandThumb2', 'mixamorig:RightHandThumb3', 'mixamorig:RightHandThumb4' # Thumb
        ,
        
            'mixamorig:LeftHand', # Wrist
            'mixamorig:LeftHandMiddle1', 'mixamorig:LeftHandMiddle2', 'mixamorig:LeftHandMiddle3', 'mixamorig:LeftHandMiddle4', # Middle
            'mixamorig:LeftHandRing1', 'mixamorig:LeftHandRing2', 'mixamorig:LeftHandRing3', 'mixamorig:LeftHandRing4', # Ring
            'mixamorig:LeftHandPinky1', 'mixamorig:LeftHandPinky2', 'mixamorig:LeftHandPinky3', 'mixamorig:LeftHandPinky4', # Pinky
            'mixamorig:LeftHandIndex1', 'mixamorig:LeftHandIndex2', 'mixamorig:LeftHandIndex3', 'mixamorig:LeftHandIndex4', # Index
            'mixamorig:LeftHandThumb1', 'mixamorig:LeftHandThumb2', 'mixamorig:LeftHandThumb3', 'mixamorig:LeftHandThumb4' # Thumb
    ], 0.2)
    o3d.visualization.draw_geometries([hand_mesh], mesh_show_wireframe=True, mesh_show_back_face=True)
    skin_mesh = SkinnableMesh(mesh_data['verts'], mesh_data['bone_cors'], mesh_data['bone_parents'], mesh_data['weight'])
    pose = torch.zeros(67, 4)
    pose[..., 0] = 1.0
    pose[10] = matrix_to_quaternion(euler_angles_to_matrix(torch.tensor([0.0, 0.0, torch.pi/4]), 'XYZ'))
    posed_verts = skin_mesh.skin(pose)
    mesh = Trimesh(vertices=posed_verts.numpy(), faces=mesh_data['faces'])
    trimesh.Scene([mesh]).show()
