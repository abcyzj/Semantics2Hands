import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_apply, quaternion_multiply


class ForwardKinematics:
    def __init__(self, parents):
        self.topology = [-1] * len(parents)
        self.rotation_map = []
        for i, parent_i in enumerate(parents):
            self.topology[i] = parent_i
            self.rotation_map.append(i)

    '''
    rotation should have shape batch_size * Time * Joint_num * (3/4)
    position should have shape batch_size * Time * 3
    offset should have shape batch_size * Joint_num * 3
    output have shape batch_size * Time * Joint_num * 3
    '''
    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', pose_repr='quaternion', world=True, ret_rot=False, ret_ori_repr=False):
        if pose_repr not in ['quaternion', 'euler', 'ortho6d']:
            raise NotImplementedError()

        if pose_repr == 'quaternion' and rotation.shape[-1] != 4:
            raise Exception('Unexpected shape of rotation')
        elif pose_repr == 'euler' and rotation.shape[-1] != 3:
            raise Exception('Unexpected shape of rotation')
        elif pose_repr == 'ortho6d' and rotation.shape[-1] != 6:
            raise Exception('Unexpected shape of rotation')

        if pose_repr == 'quaternion':
            norm = torch.norm(rotation, dim=-1, keepdim=True)
            rotation = rotation / norm


        if pose_repr == 'quaternion':
            if ret_ori_repr:
                local_transform = rotation
            else:
                local_transform = self.transform_from_quaternion(rotation)
        elif pose_repr == 'euler':
            local_transform = self.transform_from_euler(rotation, order)
        elif pose_repr == 'ortho6d':
            local_transform = self.transform_from_ortho6d(rotation)

        global_pos = [position]
        if pose_repr == 'quaternion' and ret_ori_repr:
            offset = offset.reshape(-1, 1, offset.shape[-2], offset.shape[-1])
            global_trans = [local_transform[..., 0, :]]
        else:
            offset = offset.reshape(-1, 1, offset.shape[-2], offset.shape[-1], 1)
            global_trans = [local_transform[..., 0, :, :]]
        for i, pi in enumerate(self.topology):
            if pi == -1:
                assert i == 0
                continue

            if pose_repr == 'quaternion' and ret_ori_repr:
                global_trans.append(quaternion_multiply(global_trans[pi].clone(), local_transform[..., i, :].clone()))
                global_pos.append(quaternion_apply(global_trans[pi].clone(), offset[..., i, :]))
            else:
                global_trans.append(torch.matmul(global_trans[pi].clone(), local_transform[..., i, :, :].clone()))
                global_pos.append(torch.matmul(global_trans[pi].clone(), offset[..., i, :, :]).squeeze(-1))
            if world: global_pos[i] = global_pos[i] + global_pos[pi]

        if pose_repr == 'quaternion' and ret_ori_repr:
            global_trans = torch.stack(global_trans, dim=-2)
        elif pose_repr == 'ortho6d' and ret_ori_repr:
            global_trans = torch.stack(global_trans, dim=-3)
            global_trans = self.transform_to_ortho6d(global_trans)
        else:
            global_trans = torch.stack(global_trans, dim=-3)
        result = torch.stack(global_pos, dim=-2)
        if ret_rot:
            return global_trans, result
        else:
            return result

    def from_local_to_world(self, res: torch.Tensor):
        res = res.clone()
        for i, pi in enumerate(self.topology):
            if pi == 0 or pi == -1:
                continue
            res[..., i, :] += res[..., pi, :]
        return res

    @staticmethod
    def transform_from_euler(euler_angles: torch.Tensor, convention: str):

        def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            one = torch.ones_like(angle)
            zero = torch.zeros_like(angle)

            if axis == "x":
                R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
            elif axis == "y":
                R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
            elif axis == "z":
                R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
            else:
                raise ValueError("letter must be either x, y or z.")

            return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("x", "y", "z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = [
            _axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]

        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quaternions: torch.Tensor):
        r, i, j, k = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    @staticmethod
    def transform_from_ortho6d(d6: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(d6, np.ndarray):
            d6 = torch.from_numpy(d6)
            to_numpy = True
        else:
            to_numpy = False
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        m = torch.stack((b1, b2, b3), dim=-2)
        if to_numpy:
            m = m.numpy()
        return m

    @staticmethod
    def transform_to_ortho6d(matrix: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        batch_dim = matrix.size()[:-2]
        if isinstance(matrix, torch.Tensor):
            return matrix[..., :2, :].clone().reshape(batch_dim + (6,))
        else:
            return matrix[..., :2, :].copy().reshape(batch_dim + (6,))

    @staticmethod
    def transform_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to quaternions.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).

        Returns:
            quaternions with real part first, as tensor of shape (..., 4).
        """
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
            """
            Returns torch.sqrt(torch.max(0, x))
            but with a zero subgradient where x is 0.
            """
            ret = torch.zeros_like(x)
            positive_mask = x > 0
            ret[positive_mask] = torch.sqrt(x[positive_mask])
            return ret

        q_abs = _sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = torch.stack(
            [
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)

        return quat_candidates[
            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
        ].reshape(batch_dim + (4,))


    @staticmethod
    def transform_to_euler(matrix: torch.Tensor, convention: str) -> torch.Tensor:

        def _angle_from_tan(
            axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
        ) -> torch.Tensor:


            i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
            if horizontal:
                i2, i1 = i1, i2
            even = (axis + other_axis) in ["XY", "YZ", "ZX"]
            if horizontal == even:
                return torch.atan2(data[..., i1], data[..., i2])
            if tait_bryan:
                return torch.atan2(-data[..., i2], data[..., i1])
            return torch.atan2(data[..., i2], -data[..., i1])


        def _index_from_letter(letter: str) -> int:
            if letter == "X":
                return 0
            if letter == "Y":
                return 1
            if letter == "Z":
                return 2
            raise ValueError("letter must be either X, Y or Z.")

        convention = ''.join(map(str.capitalize, convention))
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = _index_from_letter(convention[0])
        i2 = _index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            _angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            _angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)


class InverseKinematics:
    def __init__(self, rotations: torch.Tensor, positions: torch.Tensor, offset, parents, constrains):
        self.rotations = rotations
        self.rotations.requires_grad_(True)
        self.position = positions
        self.position.requires_grad_(True)

        self.parents = parents
        self.offset = offset
        self.constrains = constrains

        self.optimizer = torch.optim.Adam([self.position, self.rotations], lr=1e-3, betas=(0.9, 0.999))
        self.crit = nn.MSELoss()

    def step(self):
        self.optimizer.zero_grad()
        glb = self.forward(self.rotations, self.position, self.offset, order='', quater=True, world=True)
        loss = self.crit(glb, self.constrains)
        loss.backward()
        self.optimizer.step()
        self.glb = glb
        return loss.item()

    def tloss(self, time):
        return self.crit(self.glb[time, :], self.constrains[time, :])

    def all_loss(self):
        res = [self.tloss(t).detach().numpy() for t in range(self.constrains.shape[0])]
        return np.array(res)

    '''
        rotation should have shape batch_size * Joint_num * (3/4) * Time
        position should have shape batch_size * 3 * Time
        offset should have shape batch_size * Joint_num * 3
        output have shape batch_size * Time * Joint_num * 3
    '''

    def forward(self, rotation: torch.Tensor, position: torch.Tensor, offset: torch.Tensor, order='xyz', quater=False,
                world=True):
        '''
        if not quater and rotation.shape[-2] != 3: raise Exception('Unexpected shape of rotation')
        if quater and rotation.shape[-2] != 4: raise Exception('Unexpected shape of rotation')
        rotation = rotation.permute(0, 3, 1, 2)
        position = position.permute(0, 2, 1)
        '''
        result = torch.empty(rotation.shape[:-1] + (3,), device=position.device)

        norm = torch.norm(rotation, dim=-1, keepdim=True)
        rotation = rotation / norm

        if quater:
            transform = self.transform_from_quaternion(rotation)
        else:
            transform = self.transform_from_euler(rotation, order)

        offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

        result[..., 0, :] = position
        for i, pi in enumerate(self.parents):
            if pi == -1:
                assert i == 0
                continue

            result[..., i, :] = torch.matmul(transform[..., pi, :, :], offset[..., i, :, :]).squeeze()
            transform[..., i, :, :] = torch.matmul(transform[..., pi, :, :].clone(), transform[..., i, :, :].clone())
            if world: result[..., i, :] += result[..., pi, :]
        return result

    @staticmethod
    def transform_from_euler(rotation, order):
        rotation = rotation / 180 * math.pi
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]), transform)
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:3] + (3, 3), device=euler.device)
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_quaternion(quater: torch.Tensor):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m
