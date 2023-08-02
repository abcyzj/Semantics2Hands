import torch
from pytorch3d.transforms import matrix_to_euler_angles, quaternion_to_matrix

from utils.armatures import BaseArmature


class TBSOptimizer:
    def __init__(self, src_aramature: BaseArmature, target_armature: BaseArmature, device: torch.device = torch.device('cpu')):
        self.src_armature = src_aramature
        self.target_armature = target_armature
        self.device = device


    def optimize(self, lr: float, n_iter: int, verbose: bool = False, copy_bs_rotations: bool = True):
        if copy_bs_rotations:
            optim_params = self.src_armature.bs_rotations.detach().clone().to(self.device)
        else:
            optim_params = self.target_armature.bs_rotations.detach().clone().to(self.device)
        optim_params.requires_grad_(True)
        optimizer = torch.optim.Adam([optim_params], lr=lr)

        src_relative_finger_tbs, src_relative_palm_tbs = self.src_armature.relative_tbs_coordinates()
        src_relative_finger_tbs = src_relative_finger_tbs.detach().clone().to(self.device)
        src_relative_palm_tbs = src_relative_palm_tbs.detach().clone().to(self.device)
        adjacent_finger_js = {
            0: [1, 3],
            1: [0, 2],
            2: [1],
            3: [0],
            4: [0, 1, 2, 3]
        }
        joint_neighbor_mask = torch.zeros(20, 20, dtype=bool)
        for finger_j in range(5):
            adjacent_finger_joints = []
            for j in adjacent_finger_js[finger_j]:
                adjacent_finger_joints.extend([j*4, j*4+1, j*4+2, j*4+3])
            adjacent_finger_joints = torch.tensor(adjacent_finger_joints, dtype=torch.long, device=self.device)
            cur_finger_joints = [finger_j*4, finger_j*4+1, finger_j*4+2, finger_j*4+3]
            for cur_joint_index in cur_finger_joints:
                cur_joint_relative_tbs = src_relative_finger_tbs[..., cur_joint_index, adjacent_finger_joints, :].clone().norm(dim=-1).squeeze()
                cur_joint_neighbors = cur_joint_relative_tbs.topk(2, dim=-1, largest=False)[1]
                cur_joint_neighbors = adjacent_finger_joints[cur_joint_neighbors]
                joint_neighbor_mask[cur_joint_index, cur_joint_neighbors] = True

        for i in range(n_iter):
            optimizer.zero_grad()
            self.target_armature.bs_rotations = optim_params
            target_relative_finger_tbs, target_relative_palm_tbs = self.target_armature.relative_tbs_coordinates()
            src_finger_dir = src_relative_finger_tbs[..., joint_neighbor_mask, :] / torch.norm(src_relative_finger_tbs[..., joint_neighbor_mask, :], dim=-1, keepdim=True)
            target_finger_dir = target_relative_finger_tbs[..., joint_neighbor_mask, :] / torch.norm(target_relative_finger_tbs[..., joint_neighbor_mask, :], dim=-1, keepdim=True)
            corr_finger_dir_loss = -torch.mean(torch.sum(src_finger_dir * target_finger_dir, dim=-1))
            src_palm_dir = src_relative_palm_tbs / torch.norm(src_relative_palm_tbs, dim=-1, keepdim=True)
            target_palm_dir = target_relative_palm_tbs / torch.norm(target_relative_palm_tbs, dim=-1, keepdim=True)
            corr_palm_dir_loss = -torch.mean(torch.sum(src_palm_dir * target_palm_dir, dim=-1))
            euler_angles = matrix_to_euler_angles(quaternion_to_matrix(optim_params), 'XYZ')
            twist_loss = torch.sum(euler_angles[..., 0]**2)
            splay_loss = torch.sum(euler_angles[..., 1:, 2]**2) + torch.sum(torch.relu(euler_angles[..., [4], 0, 2].abs() - torch.pi/3)**2) + torch.sum(torch.relu(euler_angles[..., :4, 0, 2].abs() - torch.pi/18)**2)
            bend_loss = torch.sum(torch.clamp(euler_angles[..., 1], max=0)**2) + torch.sum(torch.clamp(euler_angles[..., 1] - torch.pi/2, min=0)**2)
            loss = corr_finger_dir_loss + corr_palm_dir_loss + 0.1 * twist_loss + 0.1 * splay_loss + 0.1 * bend_loss
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"iter {i:.2f} loss {loss.item():.2f} corr_palm_dir_loss: {corr_palm_dir_loss.item():.2f} corr_finger_dir_loss {corr_finger_dir_loss.item():.2f} twist_loss {twist_loss.item():.2f} splay_loss {splay_loss.item():.2f} bend_loss {bend_loss.item():.2f}")
            if -corr_palm_dir_loss.item() > 0.96 and -corr_finger_dir_loss.item() > 0.96:
                break
        self.target_armature.bs_rotations = optim_params.detach().clone()
