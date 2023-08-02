import pickle
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
from pytorch_lightning.cli import instantiate_class

from data.combined_motion import MixedData
from utils.armatures import MANOArmature, MixamoArmature
from utils.rotation_conversion import safe_matrix_to_euler_angles


class ResidualBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, dilation, padding, conv_mode, norm_type, dropout=0.2
    ):
        super().__init__()
        assert conv_mode in ['same', 'upsample', 'downsample']
        assert norm_type in ['batch', 'instance', 'none']

        if norm_type == 'batch':
            norm_cls = nn.BatchNorm1d
        elif norm_type == 'instance':
            norm_cls = nn.InstanceNorm1d
        elif norm_type == 'none':
            norm_cls = nn.Identity
        else:
            raise NotImplementedError()

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )
        self.norm1 = norm_cls(n_outputs)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout1 = nn.Dropout(dropout)

        if conv_mode == 'upsample':
            self.up2 = nn.Upsample(scale_factor=2, mode='linear')
        if conv_mode == 'downsample':
            self.down2 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )

        self.norm2 = norm_cls(n_outputs)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dropout2 = nn.Dropout(dropout)

        if conv_mode == 'same':
            self.net = nn.ModuleList([
                self.conv1, self.norm1, self.relu1, self.dropout1,
                self.conv2, self.norm2, self.relu2, self.dropout2
            ])
        elif conv_mode == 'upsample':
            self.net = nn.ModuleList([
                self.conv1, self.norm1, self.relu1, self.dropout1,
                self.up2, self.norm2, self.relu2, self.dropout2
            ])
        elif conv_mode == 'downsample':
            self.net = nn.ModuleList([
                self.conv1, self.norm2, self.relu1, self.dropout1,
                self.down2, self.norm2, self.relu2, self.dropout2
            ])
        else:
            raise NotImplementedError()

        if n_inputs == n_outputs and conv_mode == 'same':
            self.resample = None
        else:
            if conv_mode == 'same':
                self.resample = nn.Conv1d(n_inputs, n_outputs, 1)
            elif conv_mode == 'downsample':
                self.resample = nn.Sequential(
                    nn.AvgPool1d(2),
                    nn.Conv1d(n_inputs, n_outputs, 1)
                )
            else:
                self.resample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear'),
                    nn.Conv1d(n_inputs, n_outputs, 1)
                )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, rot: torch.Tensor):
        '''
        rot: (B, C_rot, T)
        offsets_repr: (B, C_out)
        return: (B, C_out, T)
        '''
        out = rot
        for layer in self.net:
            out = layer(out)
        res = rot if self.resample is None else self.resample(rot)
        return self.relu(out + res)


class StaticEncoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, n_layers: int, norm_type: str, dropout: float = 0.2):
        super().__init__()

        if norm_type == 'batch':
            norm_cls = nn.BatchNorm1d
        elif norm_type == 'instance':
            norm_cls = nn.InstanceNorm1d
        elif norm_type == 'none':
            norm_cls = nn.Identity
        else:
            raise NotImplementedError()

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Conv1d(in_channel, out_channel, 1))
            else:
                layers.append(nn.Conv1d(out_channel, out_channel, 1))
            layers.append(norm_cls(out_channel))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)


    def forward(self, static: torch.Tensor):
        return self.net(static)


class TBSRecon(nn.Module):
    def __init__(self, in_channel: int, hidden_channel: int, out_channel: int, kernel_size: int, n_layers: int, static_in_channel: int, static_hidden_channel: int, n_static_layers: int, norm_type: str):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.res_blocks = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.res_blocks.append(ResidualBlock(in_channel + static_hidden_channel, hidden_channel, kernel_size, 1, padding, 'same', norm_type, 0.0))
            elif i == n_layers - 1:
                self.res_blocks.append(ResidualBlock(hidden_channel + static_hidden_channel, out_channel, kernel_size, 1, padding, 'same', norm_type, 0.0))
            else:
                self.res_blocks.append(ResidualBlock(hidden_channel + static_hidden_channel, hidden_channel, kernel_size, 1, padding, 'same', norm_type, 0.0))

        self.static_blocks = nn.ModuleList([StaticEncoder(static_in_channel, static_hidden_channel, n_static_layers, norm_type, 0.0) for _ in range(n_layers)])


    def forward(self, rot: torch.Tensor, static: torch.Tensor):
        '''
        rot: (B, C_rot, T)
        static: (B, C_static)
        return: (B, C_out, T)
        '''
        static = static.unsqueeze(-1)
        for r_block, s_block in zip(self.res_blocks, self.static_blocks):
            static_repr = s_block(static).repeat(1, 1, rot.shape[-1])
            rot = r_block(torch.cat([rot, static_repr], dim=1))

        return rot


class TBSNet(pl.LightningModule):
    def __init__(
            self,
            hidden_channel: int,
            kernel_size: int,
            n_layers: int,
            static_hidden_channel: int,
            n_static_layers: int,
            norm_type: str,
            pose_repr: str,
            smplx_model_path: str,
            mano_axis_path: str,
            is_rhand: bool,
            semi_supervised: bool,
            optim_init: dict,
            anatomical_loss_type: str = 'euler',
            rf_mask: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            rp_mask: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
            distance_weighted: bool = False,
            lambda_rf_unweighted: float = 0.0,
            lambda_rf: float = 1.0,
            lambda_rp: float = 1.0,
            lambda_semi_rf: float = 1.0,
            lambda_semi_rp: float = 1.0,
            lambda_anatomical: float = 0.1,
            tbs_input_normalization: bool = True
    ):
        super().__init__()

        self.save_hyperparameters()

        self.pose_repr = pose_repr
        self.smplx_model_path = smplx_model_path
        self.mano_axis_path = mano_axis_path
        self.is_rhand = is_rhand
        self.semi_supervised = semi_supervised
        self.optim_init = optim_init
        self.anatomical_loss_type = anatomical_loss_type
        self.rf_mask = rf_mask
        self.rp_mask = rp_mask
        self.distance_weighted = distance_weighted
        self.lambda_rf_unweighted = lambda_rf_unweighted
        self.lambda_rf = lambda_rf
        self.lambda_rp = lambda_rp
        self.lambda_semi_rf = lambda_semi_rf
        self.lambda_semi_rp = lambda_semi_rp
        self.lambda_anatomical = lambda_anatomical
        self.tbs_input_normalization = tbs_input_normalization

        if pose_repr == 'quaternion':
            pose_dim = 4
        elif pose_repr == 'ortho6d':
            pose_dim = 6
        else:
            raise NotImplementedError()

        self.recon_nets = nn.ModuleList()
        for extra_channel, in_channel in zip([63, 0], [1740, 1750]): # mixamo, mano
            out_channel = 5 * 3 * pose_dim # 5 fingers, 3 joints, pose_dim
            static_in_channel = extra_channel + 5 * 4 * 3 * 3 # add tbs matrix
            self.recon_nets.append(TBSRecon(in_channel, hidden_channel, out_channel, kernel_size, n_layers, static_in_channel, static_hidden_channel, n_static_layers, norm_type))


    def forward(self, tbs_input: List[Optional[torch.Tensor]], static: List[Optional[torch.Tensor]]):
        '''
        tbs_input: List(B, C_rot, T)
        static: List(B, C_static)
        return: List(B, C_out, T)
        '''
        assert len(tbs_input) == len(static)

        res = []
        for tbs, s, net in zip(tbs_input, static, self.recon_nets):
            if tbs is None or s is None:
                res.append(None)
            else:
                res.append(net(tbs, s))

        return res


    def closest_neighbor(self, r_f: torch.Tensor):
        B, T = r_f.shape[:2]
        adjacent_finger_js = {
            0: [1, 3],
            1: [0, 2],
            2: [1],
            3: [0],
            4: [0, 1, 2, 3]
        }
        joint_neighbor_mask = torch.zeros(B, T, 20, 20, dtype=bool, device=r_f.device)
        for finger_j in range(5):
            adjacent_finger_joints = []
            for j in adjacent_finger_js[finger_j]:
                adjacent_finger_joints.extend([j*4, j*4+1, j*4+2, j*4+3])
            adjacent_finger_joints = torch.tensor(adjacent_finger_joints, dtype=torch.long).to(r_f.device) # (n_neighbor)
            cur_finger_joints = [finger_j*4, finger_j*4+1, finger_j*4+2, finger_j*4+3]
            for cur_joint_index in cur_finger_joints:
                cur_joint_relative_tbs = r_f[..., cur_joint_index, adjacent_finger_joints, :].clone().norm(dim=-1) # (B, T, n_neighbor)
                top_k_neighbors = cur_joint_relative_tbs.topk(self.neighbor_top_k, dim=-1, largest=False)[1] # (B, T, 2)
                top_k_neighbors = adjacent_finger_joints.repeat(B, T, 1).gather(2, top_k_neighbors) # (B, T, 2)
                joint_neighbor_mask[..., cur_joint_index, :].scatter_(2, top_k_neighbors, True)

        return joint_neighbor_mask


    def setup(self, stage: Optional[str] = None):
        with open(self.mano_axis_path, 'rb') as f:
            mano_axis = pickle.load(f)
        self.mano_armature = MANOArmature(self.is_rhand, self.smplx_model_path, mano_axis)


    def anatomical_loss(self, bs_rotation: torch.Tensor):
        if bs_rotation.shape[-1] == 4:
            bs_rotation = quaternion_to_matrix(bs_rotation)
        elif bs_rotation.shape[-1] == 6:
            bs_rotation = rotation_6d_to_matrix(bs_rotation)
        else:
            raise NotImplementedError()

        if self.anatomical_loss_type == 'euler':
            euler_angles = safe_matrix_to_euler_angles(bs_rotation, 'XYZ')
            twist_loss = torch.sum(euler_angles[..., 0]**2, dim=[2, 3]).mean()
            splay_loss = torch.sum(euler_angles[..., 1:, 2]**2, dim=[2, 3]).mean() + torch.sum(torch.relu(euler_angles[..., [4], 0, 2].abs() - torch.pi/3)**2, dim=2).mean() + torch.sum(torch.relu(euler_angles[..., :4, 0, 2].abs() - torch.pi/6)**2, dim=2).mean()
            bend_loss = torch.sum(torch.clamp(euler_angles[..., 1:, 1], max=0)**2, dim=[2, 3]).mean() + torch.sum(torch.clamp(euler_angles[..., 1] - torch.pi/2, min=0)**2, dim=[2, 3]).mean() + torch.sum(torch.clamp(euler_angles[..., :4, 0, 1] + torch.pi/18, max=0)**2, dim=2).mean() + torch.sum(torch.clamp(euler_angles[..., [4], 0, 1] + torch.pi/3, max=0)**2, dim=2).mean()
        elif self.anatomical_loss_type == 'axis':
            axis_angle = matrix_to_axis_angle(bs_rotation)
            axis = axis_angle / axis_angle.norm(dim=-1, keepdim=True)
            twist_loss = torch.sum(axis[..., 1:, 0]**2, dim=[2, 3]).mean() + torch.sum(torch.relu(axis[..., :4, 0, 0].abs() - np.cos(torch.pi / 2 - torch.pi / 36))**2, dim=[2]).mean() + torch.sum(torch.relu(axis[..., [4], 0, 0].abs() - np.cos(torch.pi / 2 - torch.pi / 3))**2, dim=[2]).mean()
            splay_loss = torch.sum(axis[..., 1:, 2]**2, dim=[2, 3]).mean() + torch.sum(torch.relu(axis[..., :4, 0, 2].abs() - np.cos(torch.pi / 2 - torch.pi / 18))**2, dim=[2]).mean() + torch.sum(torch.relu(axis[..., [4], 0, 2].abs() - np.cos(torch.pi / 2 - torch.pi / 3))**2, dim=[2]).mean()
            bend_loss = torch.sum((axis[..., 1:, 1] - 1)**2, dim=[2, 3]).mean() + torch.sum(torch.relu(1 - axis[..., :4, 0, 1] - np.cos(torch.pi / 2 - torch.pi / 9))**2, dim=[2]).mean() + torch.sum(torch.relu(1 - axis[..., [4], 0, 1] - np.cos(torch.pi / 2 - torch.pi / 3))**2, dim=[2]).mean()
            angle = axis_angle.norm(dim=-1)
            non_zero_mask = torch.abs(angle) > 1e-10
            new_angle = torch.zeros_like(angle)
            new_angle[non_zero_mask] = angle[non_zero_mask]
            bend_loss += torch.sum(torch.relu(new_angle - torch.pi / 2)**2, dim=[2, 3]).mean()
        else:
            raise NotImplementedError()

        return twist_loss, splay_loss, bend_loss


    def mano_forward(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        r_f = b['relative_finger_tbs']
        r_p = b['relative_palm_tbs']
        if self.tbs_input_normalization:
            r_f_dir = F.normalize(r_f, dim=-2).reshape(B, -1, T)
            r_p_dir = F.normalize(r_p, dim=-2).reshape(B, -1, T)
            tbs_input = torch.cat([r_f_dir, r_p_dir, shapes], dim=1)
        else:
            r_f = r_f.reshape(B, -1, T)
            r_p = r_p.reshape(B, -1, T)
            tbs_input = torch.cat([r_f, r_p, shapes], dim=1)
        static_data = m_tbs.reshape(B, -1) # only use the first frame's m_tbs for static data
        recon_bs = self([None, tbs_input], [None, static_data])[1] # (B, C_out, T)
        if self.pose_repr == 'quaternion':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 4)
        elif self.pose_repr == 'ortho6d':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 6)
        else:
            raise NotImplementedError()

        return recon_bs


    def mano_loss(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        recon_bs = self.mano_forward(b, shapes, m_tbs)
        if self.pose_repr == 'quaternion':
            bs_recon_loss = F.mse_loss(recon_bs, b['bs_rotations'].permute(0, 4, 1, 2, 3))
        elif self.pose_repr == 'ortho6d':
            gt_bs = matrix_to_rotation_6d(quaternion_to_matrix(b['bs_rotations'].permute(0, 4, 1, 2, 3)))
            bs_recon_loss = F.mse_loss(recon_bs, gt_bs)
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        recon_bs = recon_bs.reshape(B*T, 1, 5, 3, 4)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        self.mano_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, recon_r_p = self.mano_armature.relative_tbs_coordinates()
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        r_f_dir = F.normalize(r_f, dim=-1).reshape(B*T, 1, 20, 20, 3)
        r_p_dir = F.normalize(r_p, dim=-1).reshape(B*T, 1, 20, 9, 3)
        if self.distance_weighted:
            dist_weight = F.softmin(r_f.norm(dim=-1).reshape(B*T, 1, 20, 20), dim=-1).unsqueeze(-1) + self.lambda_rf_unweighted
            rf_dir_loss = -(recon_r_f_dir * r_f_dir * dist_weight)[..., self.rf_mask, :, :].sum(dim=[-1, -2]).mean()
        else:
            rf_dir_loss = -(recon_r_f_dir * r_f_dir)[..., self.rf_mask, :, :].sum(dim=-1).mean()
        rp_dir_loss = -(recon_r_p_dir * r_p_dir)[..., self.rp_mask, :].sum(dim=-1).mean() # abort the thumb palm anchor
        return bs_recon_loss, rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss


    def mano_semi_loss(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        recon_bs = self.mano_forward(b, shapes, m_tbs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        recon_bs = recon_bs.reshape(B*T, 1, 5, 3, 4)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        self.mano_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, recon_r_p = self.mano_armature.relative_tbs_coordinates()
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        r_f_dir = F.normalize(r_f, dim=-1).reshape(B*T, 1, 20, 20, 3)
        r_p_dir = F.normalize(r_p, dim=-1).reshape(B*T, 1, 20, 9, 3)
        if self.distance_weighted:
            dist_weight = F.softmin(r_f.norm(dim=-1).reshape(B*T, 1, 20, 20), dim=-1).unsqueeze(-1) + self.lambda_rf_unweighted
            rf_dir_loss = -(recon_r_f_dir * r_f_dir * dist_weight)[..., self.rf_mask, :, :].sum(dim=[-1, -2]).mean()
        else:
            rf_dir_loss = -(recon_r_f_dir * r_f_dir)[..., self.rf_mask, :, :].sum(dim=-1).mean()
        rp_dir_loss = -(recon_r_p_dir * r_p_dir)[..., self.rp_mask, :].sum(dim=-1).mean() # abort the thumb palm anchor
        return rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss


    def mano_test_loss(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        recon_bs = self.mano_forward(b, shapes, m_tbs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        recon_bs = recon_bs.reshape(B*T, 1, 5, 3, 4)
        self.mano_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, recon_r_p = self.mano_armature.relative_tbs_coordinates()
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        r_f_dir = F.normalize(r_f, dim=-1).reshape(B*T, 1, 20, 20, 3)
        r_p_dir = F.normalize(r_p, dim=-1).reshape(B*T, 1, 20, 9, 3)
        rf_dir_loss = -(recon_r_f_dir * r_f_dir)[..., self.rf_mask, :, :].sum(dim=-1).mean()
        rp_dir_loss = -(recon_r_p_dir * r_p_dir)[..., self.rp_mask, :].sum(dim=-1).mean() # abort the thumb palm anchor
        return rf_dir_loss, rp_dir_loss


    def mixamo_forward(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        r_f = b['relative_finger_tbs']
        r_p = b['relative_palm_tbs']
        if self.tbs_input_normalization:
            r_f_dir= F.normalize(r_f, dim=-2).reshape(B, -1, T)
            r_p_dir = F.normalize(r_p, dim=-2).reshape(B, -1, T)
            tbs_input = torch.cat([r_f_dir, r_p_dir], dim=1)
        else:
            r_f = r_f.reshape(B, -1, T)
            r_p = r_p.reshape(B, -1, T)
            tbs_input = torch.cat([r_f, r_p], dim=1)
        static_data = torch.cat([offset.reshape(B, -1), m_tbs.reshape(B, -1)], dim=1) # only use the first frame's m_tbs for static data, mixamo's m_tbs is inherently static
        recon_bs = self([tbs_input, None], [static_data, None])[0]

        if self.pose_repr == 'quaternion':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 4)
        elif self.pose_repr == 'ortho6d':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 6)
        else:
            raise NotImplementedError()

        return recon_bs

    def mixamo_loss(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        recon_bs = self.mixamo_forward(b, offset, m_tbs)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        if self.pose_repr == 'quaternion':
            bs_recon_loss = F.mse_loss(recon_bs, b['bs_rotations'].permute(0, 4, 1, 2, 3))
        elif self.pose_repr == 'ortho6d':
            gt_bs = matrix_to_rotation_6d(quaternion_to_matrix(b['bs_rotations'].permute(0, 4, 1, 2, 3)))
            bs_recon_loss = F.mse_loss(recon_bs, gt_bs)
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        self.mixamo_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, recon_r_p = self.mixamo_armature.relative_tbs_coordinates()
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        r_f_dir = F.normalize(r_f, dim=-1)
        r_p_dir = F.normalize(r_p, dim=-1)
        if self.distance_weighted:
            dist_weight = F.softmin(r_f.norm(dim=-1), dim=-1).unsqueeze(-1) + self.lambda_rf_unweighted
            rf_dir_loss = -(recon_r_f_dir * r_f_dir * dist_weight)[..., self.rf_mask, :, :].sum(dim=[-1, -2]).mean()
        else:
            rf_dir_loss = -(recon_r_f_dir * r_f_dir)[..., self.rf_mask, :, :].sum(dim=-1).mean()
        rp_dir_loss = -(recon_r_p_dir * r_p_dir)[..., self.rp_mask, :].sum(dim=-1).mean() # abort the thumb palm anchor

        return bs_recon_loss, rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss


    def mixamo_semi_loss(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        recon_bs = self.mixamo_forward(b, offset, m_tbs)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        self.mixamo_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, recon_r_p = self.mixamo_armature.relative_tbs_coordinates()
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        r_f_dir = F.normalize(r_f, dim=-1)
        r_p_dir = F.normalize(r_p, dim=-1)
        if self.distance_weighted:
            dist_weight = F.softmin(r_f.norm(dim=-1), dim=-1).unsqueeze(-1) + self.lambda_rf_unweighted
            rf_dir_loss = -(recon_r_f_dir * r_f_dir * dist_weight)[..., self.rf_mask, :, :].sum(dim=[-1, -2]).mean()
        else:
            rf_dir_loss = -(recon_r_f_dir * r_f_dir)[..., self.rf_mask, :, :].sum(dim=-1).mean()
        rp_dir_loss = -(recon_r_p_dir * r_p_dir)[..., self.rp_mask, :].sum(dim=-1).mean() # abort the thumb palm anchor

        return rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss


    def mixamo_test_loss(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        recon_bs = self.mixamo_forward(b, offset, m_tbs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        self.mixamo_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, recon_r_p = self.mixamo_armature.relative_tbs_coordinates()
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        r_f_dir = F.normalize(r_f, dim=-1)
        r_p_dir = F.normalize(r_p, dim=-1)
        rf_dir_loss = -(recon_r_f_dir * r_f_dir)[..., self.rf_mask, :, :].sum(dim=-1).mean()
        rp_dir_loss = -(recon_r_p_dir * r_p_dir)[..., self.rp_mask, :].sum(dim=-1).mean() # abort the thumb palm anchor
        return rf_dir_loss, rp_dir_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int):
        batch = MixedData.resolve_tensor(batch)

        if optimizer_idx == 1: # optimize mano model
            loss = 0.0

            for b in batch:
                B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
                if 'shapes' in b:
                    self.mano_armature.reset_shape(b['shapes'].permute(0, 2, 1).reshape(B*T, -1))
                    shapes = b['shapes']
                    mano_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
                    break

            for b in batch:
                if 'shapes' in b:
                    bs_recon_loss, rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mano_loss(b, shapes, mano_m_tbs)
                    self.log('train/mano_bs_recon_loss', bs_recon_loss.item())
                    self.log('train/mano_rf_dir_loss', rf_dir_loss.item())
                    self.log('train/mano_rp_dir_loss', rp_dir_loss.item())
                    self.log('train/mano_twist_loss', twist_loss.item())
                    self.log('train/mano_splay_loss', splay_loss.item())
                    self.log('train/mano_bend_loss', bend_loss.item())
                    loss += bs_recon_loss + self.lambda_rf * rf_dir_loss + self.lambda_rp * rp_dir_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

                elif 'offset' in b and self.semi_supervised:
                    rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mano_semi_loss(b, shapes, mano_m_tbs)
                    self.log('train/mano_semi_rf_dir_loss', rf_dir_loss.item())
                    self.log('train/mano_semi_rp_dir_loss', rp_dir_loss.item())
                    self.log('train/mano_semi_twist_loss', twist_loss.item())
                    self.log('train/mano_semi_splay_loss', splay_loss.item())
                    self.log('train/mano_semi_bend_loss', bend_loss.item())
                    loss += self.lambda_semi_rf * rf_dir_loss + self.lambda_semi_rp * rp_dir_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

            self.log('train/mano_loss', loss.item())
            return loss

        if optimizer_idx == 0: # optimize mano model
            loss = 0.0

            for b in batch:
                B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
                if 'offset' in b:
                    self.mixamo_armature = MixamoArmature(self.is_rhand, b['m_tbs'][..., 0].transpose(-1, -2), None, b['offset'][..., 0])
                    offset = b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) # normalize offset by middle finger to wrist distance
                    mixamo_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
                    break

            for b in batch:
                if 'offset' in b:
                    bs_recon_loss, rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mixamo_loss(b, offset, mixamo_m_tbs)
                    self.log('train/mixamo_bs_recon_loss', bs_recon_loss.item())
                    self.log('train/mixamo_rf_dir_loss', rf_dir_loss.item())
                    self.log('train/mixamo_rp_dir_loss', rp_dir_loss.item())
                    self.log('train/mixamo_twist_loss', twist_loss.item())
                    self.log('train/mixamo_splay_loss', splay_loss.item())
                    self.log('train/mixamo_bend_loss', bend_loss.item())
                    loss += bs_recon_loss + self.lambda_rf * rf_dir_loss + self.lambda_rp * rp_dir_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

                elif 'shapes' in b and self.semi_supervised:
                    rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mixamo_semi_loss(b, offset, mixamo_m_tbs)
                    self.log('train/mixamo_semi_rf_dir_loss', rf_dir_loss.item())
                    self.log('train/mixamo_semi_rp_dir_loss', rp_dir_loss.item())
                    self.log('train/mixamo_semi_twist_loss', twist_loss.item())
                    self.log('train/mixamo_semi_splay_loss', splay_loss.item())
                    self.log('train/mixamo_semi_bend_loss', bend_loss.item())
                    loss += self.lambda_semi_rf * rf_dir_loss + self.lambda_semi_rp * rp_dir_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

            self.log('train/mixamo_loss', loss.item())
            return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        batch = MixedData.resolve_tensor(batch)

        for b in batch:
            B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
            if 'shapes' in b:
                self.mano_armature.reset_shape(b['shapes'].permute(0, 2, 1).reshape(B*T, -1))
                shapes = b['shapes']
                mano_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
            elif 'offset' in b:
                self.mixamo_armature = MixamoArmature(self.is_rhand, b['m_tbs'][..., 0].transpose(-1, -2), None, b['offset'][..., 0])
                offset = b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) # normalize offset by middle finger to wrist distance
                mixamo_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix

        for b in batch:
            if 'offset' in b:
                bs_recon_loss, rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mixamo_loss(b, offset, mixamo_m_tbs)
                self.log('val/mixamo_bs_recon_err', bs_recon_loss.item())
                self.log('val/mixamo_rf_dir_err', rf_dir_loss.item())
                self.log('val/mixamo_rp_dir_err', rp_dir_loss.item())
                self.log('val/mixamo_twist_err', twist_loss.item())
                self.log('val/mixamo_splay_err', splay_loss.item())
                self.log('val/mixamo_bend_err', bend_loss.item())

                rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mano_semi_loss(b, shapes, mano_m_tbs)
                self.log('val/mano_semi_rf_dir_err', rf_dir_loss.item())
                self.log('val/mano_semi_rp_dir_err', rp_dir_loss.item())
                self.log('val/mano_semi_twist_err', twist_loss.item())
                self.log('val/mano_semi_splay_err', splay_loss.item())
                self.log('val/mano_semi_bend_err', bend_loss.item())

            elif 'shapes' in b:
                bs_recon_loss, rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mano_loss(b, shapes, mano_m_tbs)
                self.log('val/mano_bs_recon_err', bs_recon_loss.item())
                self.log('val/mano_rf_dir_err', rf_dir_loss.item())
                self.log('val/mano_rp_dir_err', rp_dir_loss.item())
                self.log('val/mano_twist_err', twist_loss.item())
                self.log('val/mano_splay_err', splay_loss.item())
                self.log('val/mano_bend_err', bend_loss.item())

                rf_dir_loss, rp_dir_loss, twist_loss, splay_loss, bend_loss = self.mixamo_semi_loss(b, offset, mixamo_m_tbs)
                self.log('val/mixamo_semi_rf_dir_err', rf_dir_loss.item())
                self.log('val/mixamo_semi_rp_dir_err', rp_dir_loss.item())
                self.log('val/mixamo_semi_twist_err', twist_loss.item())
                self.log('val/mixamo_semi_splay_err', splay_loss.item())
                self.log('val/mixamo_semi_bend_err', bend_loss.item())


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        batch = MixedData.resolve_tensor(batch)

        if 'offset' in batch[0] and 'shapes' in batch[1]:
            for b in batch:
                B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
                if 'shapes' in b:
                    self.mano_armature.reset_shape(b['shapes'].permute(0, 2, 1).reshape(B*T, -1))
                    shapes = b['shapes']
                    mano_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
                elif 'offset' in b:
                    self.mixamo_armature = MixamoArmature(self.is_rhand, b['m_tbs'][..., 0].transpose(-1, -2), None, b['offset'][..., 0])
                    offset = b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) # normalize offset by middle finger to wrist distance
                    mixamo_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix

            for b in batch:
                if 'shapes' in b:
                    rf_dir_loss, rp_dir_loss = self.mixamo_test_loss(b, offset, mixamo_m_tbs)
                    self.log('test/mano2mixamo_S_finger', -rf_dir_loss.item())
                    self.log('test/mano2mixamo_S_palm', -rp_dir_loss.item())

                if 'offset' in b:
                    rf_dir_loss, rp_dir_loss = self.mano_test_loss(b, shapes, mano_m_tbs)
                    self.log('test/mixamo2mano_S_finger', -rf_dir_loss.item())
                    self.log('test/mixamo2mano_S_palm', -rp_dir_loss.item())

        elif 'offset' in batch[0] and 'offset' in batch[1]:
            offsets = [b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) for b in batch]
            recon_bs = self.mixamo_forward(batch[1], offsets[0], batch[0]['m_tbs'][..., 0])
            if self.pose_repr == 'ortho6d':
                recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
            gt_bs = batch[0]['bs_rotations'].permute(0, 4, 1, 2, 3)
            self.mixamo_armature = MixamoArmature(self.is_rhand, batch[0]['m_tbs'][..., 0].transpose(-1, -2), None, batch[0]['offset'][..., 0])
            self.mixamo_armature.bs_rotations = recon_bs
            recon_joints = self.mixamo_armature.joints.clone()
            self.mixamo_armature.bs_rotations = gt_bs
            gt_joints = self.mixamo_armature.joints.clone()
            self.log('test/mixamo_MSE', torch.mean(torch.norm(recon_joints - gt_joints, dim=-1)**2).item())

    def configure_optimizers(self):
        optimizers = [instantiate_class(m.parameters(), self.optim_init) for m in self.recon_nets]
        return optimizers


class DMNet(pl.LightningModule):
    def __init__(
        self,
        hidden_channel: int,
        kernel_size: int,
        n_layers: int,
        static_hidden_channel: int,
        n_static_layers: int,
        norm_type: str,
        pose_repr: str,
        smplx_model_path: str,
        mano_axis_path: str,
        is_rhand: bool,
        semi_supervised: bool,
        optim_init: dict,
        anatomical_loss_type: str = 'euler',
        lambda_dm: float = 1.0,
        lambda_semi_dm: float = 1.0,
        lambda_anatomical: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pose_repr = pose_repr
        self.smplx_model_path = smplx_model_path
        self.mano_axis_path = mano_axis_path
        self.is_rhand = is_rhand
        self.semi_supervised = semi_supervised
        self.optim_init = optim_init
        self.anatomical_loss_type = anatomical_loss_type
        self.lambda_dm = lambda_dm
        self.lambda_semi_dm = lambda_semi_dm
        self.lambda_anatomical = lambda_anatomical

        if pose_repr == 'quaternion':
            pose_dim = 4
        elif pose_repr == 'ortho6d':
            pose_dim = 6
        else:
            raise NotImplementedError()

        self.recon_nets = nn.ModuleList()
        for extra_channel, in_channel in zip([63, 0], [400, 410]): # mixamo, mano
            out_channel = 5 * 3 * pose_dim # 5 fingers, 3 joints, pose_dim
            static_in_channel = extra_channel + 5 * 4 * 3 * 3 # add tbs matrix
            self.recon_nets.append(TBSRecon(in_channel, hidden_channel, out_channel, kernel_size, n_layers, static_in_channel, static_hidden_channel, n_static_layers, norm_type))


    def forward(self, x: List[Optional[torch.Tensor]], static_x: List[Optional[torch.Tensor]]):
        assert len(x) == len(self.recon_nets)
        assert len(static_x) == len(self.recon_nets)
        out = []
        for net, x, static_x in zip(self.recon_nets, x, static_x):
            if x is None:
                out.append(None)
            else:
                out.append(net(x, static_x))
        return out
    

    def setup(self, stage: Optional[str] = None):
        with open(self.mano_axis_path, 'rb') as f:
            mano_axis = pickle.load(f)
        self.mano_armature = MANOArmature(self.is_rhand, self.smplx_model_path, mano_axis)


    def anatomical_loss(self, bs_rotation: torch.Tensor):
        if bs_rotation.shape[-1] == 4:
            bs_rotation = quaternion_to_matrix(bs_rotation)
        elif bs_rotation.shape[-1] == 6:
            bs_rotation = rotation_6d_to_matrix(bs_rotation)
        else:
            raise NotImplementedError()

        if self.anatomical_loss_type == 'euler':
            euler_angles = safe_matrix_to_euler_angles(bs_rotation, 'XYZ')
            twist_loss = torch.nansum(euler_angles[..., 0]**2, dim=[2, 3]).nanmean()
            splay_loss = torch.nansum(euler_angles[..., 1:, 2]**2, dim=[2, 3]).nanmean() + torch.nansum(torch.relu(euler_angles[..., [4], 0, 2].abs() - torch.pi/3)**2, dim=2).nanmean() + torch.nansum(torch.relu(euler_angles[..., :4, 0, 2].abs() - torch.pi/18)**2, dim=2).nanmean()
            bend_loss = torch.nansum(torch.clamp(euler_angles[..., 1], max=0)**2, dim=[2, 3]).nanmean() + torch.nansum(torch.clamp(euler_angles[..., 1] - torch.pi/2, min=0)**2, dim=[2, 3]).nanmean()
        elif self.anatomical_loss_type == 'axis':
            axis_angle = matrix_to_axis_angle(bs_rotation)
            axis = axis_angle / axis_angle.norm(dim=-1, keepdim=True)
            twist_loss = torch.sum(axis[..., 1:, 0]**2, dim=[2, 3]).mean() + torch.sum(torch.relu(axis[..., :4, 0].abs() - np.cos(torch.pi / 2 - torch.pi / 36))**2, dim=[2, 3]).mean() + torch.sum(torch.relu(axis[..., [4], 0, 0].abs() - np.cos(torch.pi / 2 - torch.pi / 3))**2, dim=[2]).mean()
            splay_loss = torch.sum(axis[..., 1:, 2]**2, dim=[2, 3]).mean() + torch.sum(torch.relu(axis[..., :4, 2].abs() - np.cos(torch.pi / 2 - torch.pi / 18))**2, dim=[2, 3]).mean() + torch.sum(torch.relu(axis[..., [4], 0, 2].abs() - np.cos(torch.pi / 2 - torch.pi / 3))**2, dim=[2]).mean()
            bend_loss = torch.sum((axis[..., 1:, 1] - 1)**2, dim=[2, 3]).mean() + torch.sum(torch.relu(1 - axis[..., :4, 1] - np.cos(torch.pi / 2 - torch.pi / 9))**2, dim=[2, 3]).mean() + torch.sum(torch.relu(1 - axis[..., [4], 0, 1] - np.cos(torch.pi / 2 - torch.pi / 3))**2, dim=[2]).mean()
            angle = axis_angle.norm(dim=-1)
            non_zero_mask = torch.abs(angle) > 1e-10
            new_angle = torch.zeros_like(angle)
            new_angle[non_zero_mask] = angle[non_zero_mask]
            bend_loss += torch.sum(torch.relu(new_angle - torch.pi / 2)**2, dim=[2, 3]).mean()
        else:
            raise NotImplementedError()

        return twist_loss, splay_loss, bend_loss


    def mano_forward(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        distance_matrix = b['relative_finger_tbs'].norm(dim=-2).reshape(B, -1, T)
        tbs_input = torch.cat([distance_matrix, shapes], dim=1)
        static_data = m_tbs.reshape(B, -1) # only use the first frame's m_tbs for static data
        recon_bs = self([None, tbs_input], [None, static_data])[1] # (B, C_out, T)
        if self.pose_repr == 'quaternion':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 4)
        elif self.pose_repr == 'ortho6d':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 6)
        else:
            raise NotImplementedError()

        return recon_bs
    
    def mano_loss(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        recon_bs = self.mano_forward(b, shapes, m_tbs)
        if self.pose_repr == 'quaternion':
            bs_recon_loss = F.mse_loss(recon_bs, b['bs_rotations'].permute(0, 4, 1, 2, 3))
        elif self.pose_repr == 'ortho6d':
            gt_bs = matrix_to_rotation_6d(quaternion_to_matrix(b['bs_rotations'].permute(0, 4, 1, 2, 3)))
            bs_recon_loss = F.mse_loss(recon_bs, gt_bs)
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        recon_bs = recon_bs.reshape(B*T, 1, 5, 3, 4)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        self.mano_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, _ = self.mano_armature.relative_tbs_coordinates()
        recon_d_m = recon_r_f.norm(dim=-1)
        d_m = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3).reshape(B*T, 1, 20, 20, 3).norm(dim=-1)
        dm_loss = F.mse_loss(F.normalize(recon_d_m, dim=-1), F.normalize(d_m, dim=-1))
        return bs_recon_loss, dm_loss, twist_loss, splay_loss, bend_loss

    def mano_semi_loss(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        recon_bs = self.mano_forward(b, shapes, m_tbs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        recon_bs = recon_bs.reshape(B*T, 1, 5, 3, 4)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        self.mano_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, _ = self.mano_armature.relative_tbs_coordinates()
        recon_d_m = recon_r_f.norm(dim=-1)
        d_m = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3).reshape(B*T, 1, 20, 20, 3).norm(dim=-1)
        dm_loss = F.mse_loss(F.normalize(recon_d_m, dim=-1), F.normalize(d_m, dim=-1))

        return dm_loss, twist_loss, splay_loss, bend_loss


    def mano_test_loss(self, b: torch.Tensor, shapes: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        recon_bs = self.mano_forward(b, shapes, m_tbs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        # bs copy
        # recon_bs = b['bs_rotations'].permute(0, 4, 1, 2, 3)
        # bs copy
        recon_bs = recon_bs.reshape(B*T, 1, 5, 3, 4)
        self.mano_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        # hand copy
        # hand_rotations = b['rotations'].permute(0, 3, 1, 2).reshape(B*T, 1, 21, 4)
        # self.mano_armature.hand_rotations = F.normalize(hand_rotations, dim=-1)
        # hand copy
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        r_f_dir = F.normalize(r_f, dim=-1).reshape(B*T, 1, 20, 20, 3)
        r_p_dir = F.normalize(r_p, dim=-1).reshape(B*T, 1, 20, 9, 3)
        recon_r_f, recon_r_p = self.mano_armature.relative_tbs_coordinates()
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        rf_dir_loss = -(recon_r_f_dir * r_f_dir).sum(dim=-1).mean()
        rp_dir_loss = -(recon_r_p_dir * r_p_dir).sum(dim=-1).mean()
        return rf_dir_loss, rp_dir_loss


    def mixamo_forward(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
        distance_matrix = b['relative_finger_tbs'].norm(dim=-2).reshape(B, -1, T)
        tbs_input = distance_matrix
        static_data = torch.cat([offset.reshape(B, -1), m_tbs.reshape(B, -1)], dim=1) # only use the first frame's m_tbs for static data, mixamo's m_tbs is inherently static
        recon_bs = self([tbs_input, None], [static_data, None])[0]

        if self.pose_repr == 'quaternion':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 4)
        elif self.pose_repr == 'ortho6d':
            recon_bs = recon_bs.permute(0, 2, 1).reshape(B, T, 5, 3, 6)
        else:
            raise NotImplementedError()

        return recon_bs


    def mixamo_loss(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        recon_bs = self.mixamo_forward(b, offset, m_tbs)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        if self.pose_repr == 'quaternion':
            bs_recon_loss = F.mse_loss(recon_bs, b['bs_rotations'].permute(0, 4, 1, 2, 3))
        elif self.pose_repr == 'ortho6d':
            gt_bs = matrix_to_rotation_6d(quaternion_to_matrix(b['bs_rotations'].permute(0, 4, 1, 2, 3)))
            bs_recon_loss = F.mse_loss(recon_bs, gt_bs)
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        self.mixamo_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, _ = self.mixamo_armature.relative_tbs_coordinates()
        recon_d_m = recon_r_f.norm(dim=-1)
        d_m = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3).norm(dim=-1)
        dm_loss = F.mse_loss(F.normalize(recon_d_m, dim=-1), F.normalize(d_m, dim=-1))

        return bs_recon_loss, dm_loss, twist_loss, splay_loss, bend_loss


    def mixamo_semi_loss(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        recon_bs = self.mixamo_forward(b, offset, m_tbs)
        twist_loss, splay_loss, bend_loss = self.anatomical_loss(recon_bs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        self.mixamo_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        recon_r_f, _ = self.mixamo_armature.relative_tbs_coordinates()
        recon_d_m = recon_r_f.norm(dim=-1)
        d_m = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3).norm(dim=-1)
        dm_loss = F.mse_loss(F.normalize(recon_d_m, dim=-1), F.normalize(d_m, dim=-1))

        return dm_loss, twist_loss, splay_loss, bend_loss
    

    def mixamo_test_loss(self, b: torch.Tensor, offset: torch.Tensor, m_tbs: torch.Tensor):
        recon_bs = self.mixamo_forward(b, offset, m_tbs)
        if self.pose_repr == 'ortho6d':
            recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
        # bs copy
        # recon_bs = b['bs_rotations'].permute(0, 4, 1, 2, 3)
        # bs copy
        self.mixamo_armature.bs_rotations = F.normalize(recon_bs, dim=-1)
        # hand copy
        # hand_rotations = b['rotations'].permute(0, 3, 1, 2)
        # self.mixamo_armature.hand_rotations = F.normalize(hand_rotations, dim=-1)
        # hand copy
        r_f, r_p = b['relative_finger_tbs'].permute(0, 4, 1, 2, 3), b['relative_palm_tbs'].permute(0, 4, 1, 2, 3)
        recon_r_f, recon_r_p = self.mixamo_armature.relative_tbs_coordinates()
        r_f_dir = F.normalize(r_f, dim=-1)
        r_p_dir = F.normalize(r_p, dim=-1)
        recon_r_f_dir = F.normalize(recon_r_f, dim=-1)
        recon_r_p_dir = F.normalize(recon_r_p, dim=-1)
        rf_dir_loss = -(r_f_dir * recon_r_f_dir).sum(dim=-1).mean()
        rp_dir_loss = -(r_p_dir * recon_r_p_dir).sum(dim=-1).mean()
        return rf_dir_loss, rp_dir_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int):
        batch = MixedData.resolve_tensor(batch)

        if optimizer_idx == 1: # optimize mano model
            loss = 0.0

            for b in batch:
                B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
                if 'shapes' in b:
                    self.mano_armature.reset_shape(b['shapes'].permute(0, 2, 1).reshape(B*T, -1))
                    shapes = b['shapes']
                    mano_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
                    break

            for b in batch:
                if 'shapes' in b:
                    bs_recon_loss, dm_loss, twist_loss, splay_loss, bend_loss = self.mano_loss(b, shapes, mano_m_tbs)
                    self.log('train/mano_bs_recon_loss', bs_recon_loss.item())
                    self.log('train/mano_dm_loss', dm_loss.item())
                    self.log('train/mano_twist_loss', twist_loss.item())
                    self.log('train/mano_splay_loss', splay_loss.item())
                    self.log('train/mano_bend_loss', bend_loss.item())
                    loss += bs_recon_loss + self.lambda_dm * dm_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

                elif 'offset' in b and self.semi_supervised:
                    dm_loss, twist_loss, splay_loss, bend_loss = self.mano_semi_loss(b, shapes, mano_m_tbs)
                    self.log('train/mano_semi_dm_loss', dm_loss.item())
                    self.log('train/mano_semi_twist_loss', twist_loss.item())
                    self.log('train/mano_semi_splay_loss', splay_loss.item())
                    self.log('train/mano_semi_bend_loss', bend_loss.item())
                    loss += self.lambda_semi_dm * dm_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

            self.log('train/mano_loss', loss.item())
            return loss

        if optimizer_idx == 0: # optimize mano model
            loss = 0.0

            for b in batch:
                B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
                if 'offset' in b:
                    self.mixamo_armature = MixamoArmature(self.is_rhand, b['m_tbs'][..., 0].transpose(-1, -2), None, b['offset'][..., 0])
                    offset = b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) # normalize offset by middle finger to wrist distance
                    mixamo_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
                    break


            for b in batch:
                if 'offset' in b:
                    bs_recon_loss, dm_loss, twist_loss, splay_loss, bend_loss = self.mixamo_loss(b, offset, mixamo_m_tbs)
                    self.log('train/mixamo_bs_recon_loss', bs_recon_loss.item())
                    self.log('train/mixamo_dm_loss', dm_loss.item())
                    self.log('train/mixamo_twist_loss', twist_loss.item())
                    self.log('train/mixamo_splay_loss', splay_loss.item())
                    self.log('train/mixamo_bend_loss', bend_loss.item())
                    loss += bs_recon_loss + self.lambda_dm * dm_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

                elif 'shapes' in b and self.semi_supervised:
                    dm_loss, twist_loss, splay_loss, bend_loss = self.mixamo_semi_loss(b, offset, mixamo_m_tbs)
                    self.log('train/mixamo_semi_dm_loss', dm_loss.item())
                    self.log('train/mixamo_semi_twist_loss', twist_loss.item())
                    self.log('train/mixamo_semi_splay_loss', splay_loss.item())
                    self.log('train/mixamo_semi_bend_loss', bend_loss.item())
                    loss += self.lambda_semi_dm * dm_loss + self.lambda_anatomical * (twist_loss + splay_loss + bend_loss)

            self.log('train/mixamo_loss', loss.item())
            return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        batch = MixedData.resolve_tensor(batch)

        for b in batch:
            B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
            if 'shapes' in b:
                self.mano_armature.reset_shape(b['shapes'].permute(0, 2, 1).reshape(B*T, -1))
                shapes = b['shapes']
                mano_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
            elif 'offset' in b:
                self.mixamo_armature = MixamoArmature(self.is_rhand, b['m_tbs'][..., 0].transpose(-1, -2), None, b['offset'][..., 0])
                offset = b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) # normalize offset by middle finger to wrist distance
                mixamo_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix

        for b in batch:
            if 'offset' in b:
                bs_recon_loss, dm_loss, twist_loss, splay_loss, bend_loss = self.mixamo_loss(b, offset, mixamo_m_tbs)
                self.log('val/mixamo_bs_recon_err', bs_recon_loss.item())
                self.log('val/mixamo_dm_err', dm_loss.item())
                self.log('val/mixamo_twist_err', twist_loss.item())
                self.log('val/mixamo_splay_err', splay_loss.item())
                self.log('val/mixamo_bend_err', bend_loss.item())

                dm_loss, twist_loss, splay_loss, bend_loss = self.mano_semi_loss(b, shapes, mano_m_tbs)
                self.log('val/mano_semi_dm_err', dm_loss.item())
                self.log('val/mano_semi_twist_err', twist_loss.item())
                self.log('val/mano_semi_splay_err', splay_loss.item())
                self.log('val/mano_semi_bend_err', bend_loss.item())

            elif 'shapes' in b:
                bs_recon_loss, dm_loss, twist_loss, splay_loss, bend_loss = self.mano_loss(b, shapes, mano_m_tbs)
                self.log('val/mano_bs_recon_err', bs_recon_loss.item())
                self.log('val/mano_dm_err', dm_loss.item())
                self.log('val/mano_twist_err', twist_loss.item())
                self.log('val/mano_splay_err', splay_loss.item())
                self.log('val/mano_bend_err', bend_loss.item())

                dm_loss, twist_loss, splay_loss, bend_loss = self.mixamo_semi_loss(b, offset, mixamo_m_tbs)
                self.log('val/mixamo_semi_dm_err', dm_loss.item())
                self.log('val/mixamo_semi_twist_err', twist_loss.item())
                self.log('val/mixamo_semi_splay_err', splay_loss.item())
                self.log('val/mixamo_semi_bend_err', bend_loss.item())


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        batch = MixedData.resolve_tensor(batch)

        if 'offset' in batch[0] and 'shapes' in batch[1]:
            for b in batch:
                B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
                if 'shapes' in b:
                    self.mano_armature.reset_shape(b['shapes'].permute(0, 2, 1).reshape(B*T, -1))
                    shapes = b['shapes']
                    mano_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix
                elif 'offset' in b:
                    self.mixamo_armature = MixamoArmature(self.is_rhand, b['m_tbs'][..., 0].transpose(-1, -2), None, b['offset'][..., 0])
                    offset = b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) # normalize offset by middle finger to wrist distance
                    mixamo_m_tbs = b['m_tbs'][..., 0] # only use first frame's tbs matrix

            for b in batch:
                if 'offset' in b:
                    rf_loss, rp_loss = self.mano_test_loss(b, shapes, mano_m_tbs)
                    self.log('test/mixamo2mano_S_finger', -rf_loss.item())
                    self.log('test/mixamo2mano_S_palm', -rp_loss.item())

                elif 'shapes' in b:
                    rf_loss, rp_loss = self.mixamo_test_loss(b, offset, mixamo_m_tbs)
                    self.log('test/mano2mixamo_S_finger', -rf_loss.item())
                    self.log('test/mano2mixamo_S_palm', -rp_loss.item())

        elif 'offset' in batch[0] and 'offset' in batch[1]:
            offsets = [b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) for b in batch]
            recon_bs = self.mixamo_forward(batch[1], offsets[0], batch[0]['m_tbs'][..., 0])
            if self.pose_repr == 'ortho6d':
                recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
            gt_bs = batch[0]['bs_rotations'].permute(0, 4, 1, 2, 3)
            self.mixamo_armature = MixamoArmature(self.is_rhand, batch[0]['m_tbs'][..., 0].transpose(-1, -2), None, batch[0]['offset'][..., 0])

            # bs copy
            # recon_bs = batch[1]['bs_rotations'].permute(0, 4, 1, 2, 3)
            # bs copy

            self.mixamo_armature.bs_rotations = recon_bs

            # hand copy
            # hand_rotations = batch[1]['rotations'].permute(0, 3, 1, 2)
            # self.mixamo_armature.hand_rotations = F.normalize(hand_rotations, dim=-1)
            # hand copy

            recon_joints = self.mixamo_armature.joints.clone()
            self.mixamo_armature.bs_rotations = gt_bs
            gt_joints = self.mixamo_armature.joints.clone()
            self.log('test/mixamo_MSE', torch.mean(torch.norm(recon_joints - gt_joints, dim=-1)**2).item())

    def configure_optimizers(self):
        optimizers = [instantiate_class(m.parameters(), self.optim_init) for m in self.recon_nets]
        return optimizers
