import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import pyglet
import trimesh
from trimesh import Trimesh
from trimesh.geometry import weighted_vertex_normals
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh.triangles import points_to_barycentric
from trimesh.viewer.windowed import SceneViewer

from utils.armatures import MixamoArmature
from utils.mesh import extract_joint_mesh, load_path_from_axis
from utils.armature_config import config_mixamo_armature


class AxisViewer(SceneViewer):
    def __init__(self, scene: trimesh.Scene, origin: np.ndarray, directions: np.ndarray, cur_splay_index: int, **kwargs):
        super().__init__(scene, start_loop=False, smooth=False, **kwargs)
        self.scene = scene
        self.origin = origin
        self.directions = directions
        self.cur_splay_index = cur_splay_index % len(directions)
        self.update_bend_path()
        pyglet.app.run()

    def update_bend_path(self):
        self.cur_splay_index = self.cur_splay_index % len(self.directions)
        splay_axis = self.directions[self.cur_splay_index]
        splay_path = load_path_from_axis(splay_axis, self.origin, whole_axis=False)
        splay_path.colors = np.tile(np.array([255, 0, 0, 255]), (len(splay_path.entities), 1))
        bend_axis = self.directions[(self.cur_splay_index + len(self.directions)//4) % len(self.directions)]
        bend_path = load_path_from_axis(bend_axis, self.origin, whole_axis=False)
        self.scene.delete_geometry('bend_path')
        self.scene.delete_geometry('splay_path')
        self.scene.add_geometry(bend_path, geom_name='bend_path')
        self.scene.add_geometry(splay_path, geom_name='splay_path')
        self._update_vertex_list()
        self._update_perspective(self.width, self.height)
        self.on_draw()
        print(f'Current splay axis: {self.cur_splay_index}, {splay_axis}')

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.COMMA:
            self.cur_splay_index = (self.cur_splay_index - 1) % len(self.directions)
            self.update_bend_path()
        elif symbol == pyglet.window.key.PERIOD:
            self.cur_splay_index = (self.cur_splay_index + 1) % len(self.directions)
            self.update_bend_path()
        else:
            super().on_key_press(symbol, modifiers)

    def get_axes(self):
        splay_axis = self.directions[self.cur_splay_index]
        bend_axis = self.directions[(self.cur_splay_index + len(self.directions)//4) % len(self.directions)]
        return bend_axis, splay_axis


def bary_point_normals(mesh: Trimesh, points: np.ndarray, index_tri: int, vertex_normals: np.ndarray):
    triangles = mesh.faces[index_tri] # (n_tri, 3)
    triangle_vertices = mesh.vertices[triangles] # (n_tri, 3, 3)
    bary_coords = points_to_barycentric(triangle_vertices, points, method='cross') # (n_tri, 3)
    triangle_vertex_normals = vertex_normals[triangles] # (n_tri, 3, 3)
    point_normals = np.sum(bary_coords[:, :, np.newaxis] * triangle_vertex_normals, axis=1) # (n_tri, 3)
    return trimesh.util.unitize(point_normals)


def estimate_axis_from_mesh(mesh: Trimesh, joint_cors: np.ndarray, twist_axis: np.ndarray):
    vertex_normals = weighted_vertex_normals(len(mesh.vertices), mesh.faces, mesh.face_normals, mesh.face_angles)
    intersector = RayMeshIntersector(mesh)
    N = 360
    origins = np.tile(joint_cors, (N, 1))
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    plane_x = np.cross(twist_axis, np.array([0, 1, 0]))
    plane_x = plane_x / np.linalg.norm(plane_x)
    plane_y = np.cross(twist_axis, plane_x)
    plane_y = plane_y / np.linalg.norm(plane_y)
    directions = np.cos(angles)[:, None] * plane_x + np.sin(angles)[:, None] * plane_y
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    locations, index_ray, index_tri = intersector.intersects_location(origins, directions, multiple_hits=False)
    if len(index_ray) < N:
        raise ValueError('Not all rays hit the mesh')
    finger_diameter = np.sum(directions[:N//2][:, np.newaxis] * locations[np.newaxis], axis=2)
    finger_diameter = finger_diameter.max(axis=1) - finger_diameter.min(axis=1)

    M = 30
    possible_splay_indices = np.linspace(finger_diameter.argmax() - M//2 + N//4, finger_diameter.argmax() + M//2 + N//4, M + 1, dtype=int) % N
    possible_splay_indices = np.concatenate([possible_splay_indices, possible_splay_indices + N//2], axis=0) % N
    possible_splay_normals = bary_point_normals(mesh, locations[possible_splay_indices], index_tri[possible_splay_indices], vertex_normals)
    product = np.sum(possible_splay_normals * directions[possible_splay_indices], axis=1)
    return directions, possible_splay_indices[product.argmax()]


def joint_axes_from_mesh(mesh: o3d.geometry.TriangleMesh, joint_cors: np.ndarray, twist_axis: np.ndarray, preset_axes: Optional[np.ndarray] = None):
    mesh = Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
    twist_axis = twist_axis / np.linalg.norm(twist_axis)

    directions, splay_index = estimate_axis_from_mesh(mesh, joint_cors, twist_axis)

    if preset_axes is not None:
        if len(preset_axes.shape) == 3:
            preset_splay_axis = preset_axes[2, 2] # pick the last non-tip joint
        else:
            preset_splay_axis = preset_axes[2]
        splay_index = np.sum(directions * preset_splay_axis[np.newaxis], axis=1).argmax()

    scene = trimesh.Scene(mesh)
    axis_viewer = AxisViewer(scene, joint_cors, directions, splay_index)
    bend_axis, splay_axis = axis_viewer.get_axes()
    bend_axis = np.cross(splay_axis, twist_axis)

    return np.stack([twist_axis, bend_axis, splay_axis], axis=0)


def main(args):
    mesh_data = np.load(args.mesh_data, allow_pickle=True)
    if args.preset is not None:
        with open(args.preset, 'rb') as f:
            preset_axes = pickle.load(f)
    elif Path(args.output).exists():
        with open(args.output, 'rb') as f:
            preset_axes = pickle.load(f)
    else:
        preset_axes = None

    hand_axes = {}
    config_mixamo_armature()
    for is_rhand, hand_label in zip([True, False], ['rhand', 'lhand']):
        armature = MixamoArmature(is_rhand=is_rhand, mesh_data=mesh_data)
        mesh = extract_joint_mesh(mesh_data, armature.hand_j_labels)
        finger_tbs_axes = []
        for finger_idx, finger_labels in enumerate(armature.finger_names):
            finger_joint_indices = [mesh_data['vgrp_label'].tolist().index(l) for l in finger_labels]
            finger_joint_cors = mesh_data['vgrp_cors'][finger_joint_indices]
            twist_axis = finger_joint_cors[3] - finger_joint_cors[2]
            finger_preset_axes = None if preset_axes is None else preset_axes[hand_label][finger_idx]
            single_joint_axes = joint_axes_from_mesh(mesh, finger_joint_cors[2], twist_axis, finger_preset_axes)
            cur_finger_twist_axes = finger_joint_cors[1:] - finger_joint_cors[:-1]
            cur_finger_twist_axes = np.concatenate([cur_finger_twist_axes, cur_finger_twist_axes[-1:]], axis=0)
            cur_finger_twist_axes = cur_finger_twist_axes / np.linalg.norm(cur_finger_twist_axes, axis=1, keepdims=True)
            cur_finger_bend_axes = np.tile(single_joint_axes[1], (finger_joint_cors.shape[0], 1))
            cur_finger_bend_axes = cur_finger_bend_axes - np.sum(cur_finger_bend_axes * cur_finger_twist_axes, axis=1, keepdims=True) * cur_finger_twist_axes
            cur_finger_bend_axes = cur_finger_bend_axes / np.linalg.norm(cur_finger_bend_axes, axis=1, keepdims=True)
            cur_finger_splay_axes = np.cross(cur_finger_twist_axes, cur_finger_bend_axes)
            finger_tbs_axes.append(np.stack([cur_finger_twist_axes, cur_finger_bend_axes, cur_finger_splay_axes], axis=1))
        finger_tbs_axes = np.stack(finger_tbs_axes, axis=0)
        hand_axes[hand_label] = finger_tbs_axes

    with open(args.output, 'wb') as f:
        pickle.dump(hand_axes, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mesh_data', type=str, required=True)
    parser.add_argument('--preset', type=str, required=False)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    main(args)
