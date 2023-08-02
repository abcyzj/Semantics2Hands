import numpy as np
import torch

import utils.BVH_mod as BVH
from data.bvh_writer import write_bvh
from model.kinematics import ForwardKinematics
from utils.Quaternions import Quaternions
from utils.armatures import MixamoArmature, MANOArmature, BaseArmature

"""
1.
Specify the joints that you want to use in training and test. Other joints will be discarded.
Please start with root joint, then left leg chain, right leg chain, head chain, left shoulder chain and right shoulder chain.
See the examples below.
"""
corps_name_mixamo_hand = ['RightHand', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4']
corps_name_mano_hand = ['right_wrist', 'right_middle1', 'right_middle2', 'right_middle3', 'right_middle4', 'right_ring1', 'right_ring2', 'right_ring3', 'right_ring4', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_pinky4', 'right_index1', 'right_index2', 'right_index3', 'right_index4', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_thumb4']

"""
2.
Specify five end effectors' name.
Please follow the same order as in 1.
"""
ee_name_mixamo_hand = ['RightHandMiddle4', 'RightHandRing4', 'RightHandPinky4', 'RightHandIndex4', 'RightHandThumb4']
ee_name_mano_hand = ['right_middle4', 'right_ring4', 'right_pinky4', 'right_index4', 'right_thumb4']


"""
3.
Add previously added corps_name and ee_name at the end of the two above lists.
"""
corps_names = [corps_name_mixamo_hand, corps_name_mano_hand]
ee_names = [ee_name_mixamo_hand, ee_name_mano_hand]


class BVH_file:
    def __init__(self, file_path: str, new_root=None):
        self.anim, self._names, self.frametime = BVH.load(file_path)
        if new_root is not None:
            self.set_new_root(new_root)
        self.skeleton_type = -1
        self.edges = []
        self.edge_mat = []
        self.edge_num = 0
        self._topology = None
        self.ee_length = []

        for i, name in enumerate(self._names):
            if ':' in name:
                name = name[name.find(':') + 1:]
                self._names[i] = name

        full_fill = [1] * len(corps_names)
        for i, ref_names in enumerate(corps_names):
            for ref_name in ref_names:
                if ref_name not in self._names:
                    full_fill[i] = 0
                    break

        """
        4. 
        Here, you need to assign self.skeleton_type the corresponding index of your own dataset in corps_names or ee_names list.
        You can use self._names, which contains the joints name in original bvh file, to write your own if statement.
        """
        if full_fill[0] == 1:
            self.skeleton_type = 0
            self.set_new_root(0)
        elif full_fill[1] == 1:
            self.skeleton_type = 1
        else:
            self.skeleton_type = -1

        if self.skeleton_type == -1:
            print(self._names)
            raise Exception('Unknown skeleton')

        self.details = [i for i, name in enumerate(self._names) if name not in corps_names[self.skeleton_type]]
        self.joint_num = self.anim.shape[1]
        self.corps = []
        self.simplified_name = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        for name in corps_names[self.skeleton_type]:
            for j in range(self.anim.shape[1]):
                if name == self._names[j]:
                    self.corps.append(j)
                    break

        if len(self.corps) != len(corps_names[self.skeleton_type]):
            for i in self.corps: print(self._names[i], end=' ')
            print(self.corps, self.skeleton_type, len(self.corps), sep='\n')
            raise Exception('Problem in file', file_path)

        self.ee_id = []
        for i in ee_names[self.skeleton_type]:
            self.ee_id.append(corps_names[self.skeleton_type].index(i))

        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1
        for i in range(self.anim.shape[1]):
            if i in self.details:
                self.simplify_map[i] = -1

    def scale(self, alpha):
        self.anim.offsets *= alpha
        global_position = self.anim.positions[:, 0, :]
        global_position[1:, :] *= alpha
        global_position[1:, :] += (1 - alpha) * global_position[0, :]

    def rotate(self, theta, axis):
        q = Quaternions(np.hstack((np.cos(theta/2), np.sin(theta/2) * axis)))
        position = self.anim.positions[:, 0, :].copy()
        rotation = self.anim.rotations[:, 0, :]
        position[1:, ...] -= position[0:-1, ...]
        q_position = Quaternions(np.hstack((np.zeros((position.shape[0], 1)), position)))
        q_rotation = Quaternions.from_euler(np.radians(rotation))
        q_rotation = q * q_rotation
        q_position = q * q_position * (-q)
        self.anim.rotations[:, 0, :] = np.degrees(q_rotation.euler())
        position = q_position.imaginaries
        for i in range(1, position.shape[0]):
            position[i] += position[i-1]
        self.anim.positions[:, 0, :] = position

    @property
    def topology(self):
        if self._topology is None:
            self._topology = self.anim.parents[self.corps].copy()
            for i in range(self._topology.shape[0]):
                if i >= 1: self._topology[i] = self.simplify_map[self._topology[i]]
            self._topology = tuple(self._topology)
        return self._topology

    def get_ee_id(self):
        return self.ee_id

    def to_numpy(self, armature: BaseArmature):
        rotations = self.anim.rotations[:, self.corps, :]
        # for hand, set the rotation of the root to 0
        rotations[..., 0, :] = 0.0

        rotations = Quaternions.from_euler(np.radians(rotations)).qs

        root_name = corps_names[self.skeleton_type][0]
        if root_name in ['right_wrist', 'left_wrist']:
            assert isinstance(armature, MANOArmature)
            T = rotations.shape[0]
            hand_rotations = torch.from_numpy(rotations).unsqueeze(1).to(torch.float) # (B/T, 1, J, 4)
            armature.hand_rotations = hand_rotations
            bs_rotations = armature.bs_rotations # (B/T, 1, 5, 3, 4)
            relative_finger_tbs, relative_palm_tbs = armature.relative_tbs_coordinates() #(B/T, 1, 20, 20, 3)
            rotations = rotations.reshape(T, -1)
            m_tbs = armature.m_tbs.reshape(T, -1).numpy()
            # 10, 180, 84, 60, 1200, 540
            seq = np.concatenate([armature.shape.numpy(), m_tbs, rotations, bs_rotations.reshape(T, -1).numpy(), relative_finger_tbs.reshape(T, -1).numpy(), relative_palm_tbs.reshape(T, -1).numpy()], axis=-1)
        elif root_name in ['RightHand', 'LeftHand']:
            assert isinstance(armature, MixamoArmature)
            T = rotations.shape[0]
            hand_rotations = torch.from_numpy(rotations).unsqueeze(0).to(torch.float) # (1, T, J, 4)
            armature.hand_rotations = hand_rotations
            bs_rotations = armature.bs_rotations # (1, T, 5, 3, 4)
            relative_finger_tbs, relative_palm_tbs = armature.relative_tbs_coordinates() #(1, T, 20, 20, 3)
            rotations = rotations.reshape(T, -1)
            offsets = armature.hand_offsets.reshape(1, -1).repeat(T, 1).cpu().numpy()
            m_tbs = armature.m_tbs.reshape(1, -1).repeat(T, 1).numpy()
            # 63, 180, 84, 60, 1200, 540
            seq = np.concatenate([offsets, m_tbs, rotations, bs_rotations.reshape(T, -1).numpy(), relative_finger_tbs.reshape(T, -1).numpy(), relative_palm_tbs.reshape(T, -1).numpy()], axis=-1)

        return seq

    def to_tensor(self, armature: BaseArmature):
        res = self.to_numpy(armature)
        res = torch.tensor(res, dtype=torch.float)
        res = res.permute(1, 0)
        res = res.reshape((-1, res.shape[-1]))
        return res

    def get_position(self):
        positions = self.anim.positions
        positions = positions[:, self.corps, :]
        return positions

    @property
    def offset(self):
        return self.anim.offsets[self.corps]

    @property
    def names(self):
        return self.simplified_name

    def get_height(self):
        offset = self.offset
        topo = self.topology

        res = 0
        p = self.ee_id[0]
        while p != 0:
            res += np.dot(offset[p], offset[p]) ** 0.5 # middle finger to wrist
            p = topo[p]

        return res

    def write(self, file_path):
        motion = self.to_numpy(quater=False, edge=False)
        rotations = motion[..., :-3].reshape(motion.shape[0], -1, 3)
        positions = motion[..., -3:]
        write_bvh(self.topology, self.offset, rotations, positions, self.names, 1.0/30, 'xyz', file_path)

    def get_ee_length(self):
        if len(self.ee_length): return self.ee_length
        degree = [0] * len(self.topology)
        for i in self.topology:
            if i < 0: continue
            degree[i] += 1

        for j in self.ee_id:
            length = 0
            while degree[j] <= 1:
                t = self.offset[j]
                length += np.dot(t, t) ** 0.5
                j = self.topology[j]

            self.ee_length.append(length)

        height = self.get_height()
        ee_group = [[0, 1], [2], [3, 4]]
        for group in ee_group:
            maxv = 0
            for j in group:
                maxv = max(maxv, self.ee_length[j])
            for j in group:
                self.ee_length[j] *= height / maxv

        return self.ee_length

    def set_new_root(self, new_root):
        euler = torch.tensor(self.anim.rotations[:, 0, :], dtype=torch.float)
        transform = ForwardKinematics.transform_from_euler(euler, 'xyz')
        offset = torch.tensor(self.anim.offsets[new_root], dtype=torch.float)
        new_pos = torch.matmul(transform, offset)
        new_pos = new_pos.numpy() + self.anim.positions[:, 0, :]
        self.anim.offsets[0] = -self.anim.offsets[new_root]
        self.anim.offsets[new_root] = np.zeros((3, ))
        self.anim.positions[:, new_root, :] = new_pos
        rot0 = Quaternions.from_euler(np.radians(self.anim.rotations[:, 0, :]), order='xyz')
        rot1 = Quaternions.from_euler(np.radians(self.anim.rotations[:, new_root, :]), order='xyz')
        new_rot1 = rot0 * rot1
        new_rot0 = (-rot1)
        new_rot0 = np.degrees(new_rot0.euler())
        new_rot1 = np.degrees(new_rot1.euler())
        self.anim.rotations[:, 0, :] = new_rot0
        self.anim.rotations[:, new_root, :] = new_rot1

        new_seq = []
        vis = [0] * self.anim.rotations.shape[1]
        new_idx = [-1] * len(vis)
        new_parent = [0] * len(vis)

        def relabel(x):
            nonlocal new_seq, vis, new_idx, new_parent
            new_idx[x] = len(new_seq)
            new_seq.append(x)
            vis[x] = 1
            for y in range(len(vis)):
                if not vis[y] and (self.anim.parents[x] == y or self.anim.parents[y] == x):
                    relabel(y)
                    new_parent[new_idx[y]] = new_idx[x]

        relabel(new_root)
        self.anim.rotations = self.anim.rotations[:, new_seq, :]
        self.anim.offsets = self.anim.offsets[new_seq]
        names = self._names.copy()
        for i, j in enumerate(new_seq):
            self._names[i] = names[j]
        self.anim.parents = np.array(new_parent, dtype=int)
