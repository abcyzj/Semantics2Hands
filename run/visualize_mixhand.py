import pickle
from pathlib import Path

import numpy as np
import torch
from imageio import imwrite
from pytorch3d.transforms import matrix_to_quaternion, rotation_6d_to_matrix
from pytorch_lightning.cli import LightningArgumentParser
from tqdm import tqdm

from data.combined_motion import MixedData, MixedDataModule
from model.tbs import TBSNet
from run.train_mixhand import TBSCLI
from utils.armature_config import config_mixamo_armature
from utils.armatures import (BaseArmature, MANOArmature, MixamoArmature,
                             export_armature_animation, show_armature)


def add_alpha_channel(img: np.ndarray):
    alpha = np.ones_like(img[..., :1]) * 255
    alpha[img.sum(axis=-1) == 255*3] = 0
    return np.concatenate([img, alpha], axis=-1)

def export_img(armature: BaseArmature, color: str, output_dir: Path, name: str):
    for r in [180]:
        img = show_armature(armature, color=color, roll=r)
        img = add_alpha_channel(img)
        imwrite(output_dir / f'{name}_{r}.png', img)


class VisualizeCLI(TBSCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--ckpt_path', type=str, required=True)
        parser.add_argument('--output_dir', type=str, required=True)

    def before_instantiate_classes(self):
        self.config.data['window_size'] = 20
        super().before_instantiate_classes()


if __name__ == '__main__':
    config_mixamo_armature()
    cli = VisualizeCLI(TBSNet, MixedDataModule, run=False)
    cli.model: TBSNet = TBSNet.load_from_checkpoint(cli.config.ckpt_path, map_location=cli.model.device)
    cli.model.freeze()
    cli.datamodule.setup('test')
    cli.model.setup('test')
    dataloader = cli.datamodule.test_dataloader()

    output_dir = Path(cli.config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    mixamo_aramatures = {}
    mixamo_characters = cli.datamodule.mixdata_test.datasets_groups[0]
    for c in mixamo_characters:
        mesh_data_f = Path(cli.config.data.data_path) / 'finger_data' / f'{c}_mesh_data.npz'
        axis_data_f = Path(cli.config.data.data_path) / 'finger_data' / f'{c}_finger_axis.pkl'
        mesh_data = np.load(mesh_data_f)
        with open(axis_data_f, 'rb') as f:
            axis_data = pickle.load(f)
        mixamo_aramatures[c] = MixamoArmature(cli.config.model.is_rhand, axis_data, mesh_data)
    with open(cli.config.model.mano_axis_path, 'rb') as f:
            mano_axis = pickle.load(f)
    mano_armature = MANOArmature(cli.config.model.is_rhand, cli.config.model.smplx_model_path, mano_axis)

    for b_idx, batch in enumerate(dataloader):
        batch = MixedData.resolve_tensor(batch)
        for b in batch:
            B, T = b['m_tbs'].shape[0], b['m_tbs'].shape[-1]
            if 'shapes' in b:
                cli.model.mano_armature.reset_shape(b['shapes'].permute(0, 2, 1).reshape(B*T, -1))
                shapes = b['shapes']
                mano_m_tbs = b['m_tbs'][..., 0] # only use the first frame's tbs matrix
            elif 'offset' in b:
                offset = b['offset'][..., 0] / b['offset'][:, [1], :, 0].norm(dim=-1, keepdim=True) # normalize offset by middle finger to wrist distance
                mixamo_m_tbs = b['m_tbs'][..., 0] # only use the first frame's tbs matrix

        for b in batch:
            if 'shapes' in b:
                recon_bs = cli.model.mixamo_forward(b, offset, mixamo_m_tbs)
                if cli.model.pose_repr == 'ortho6d':
                    recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
                for s_idx in tqdm(range(recon_bs.shape[0]), desc=f'Exporting batch {b_idx}'):
                    if s_idx % 10 != 0:
                        continue
                    character_name = mixamo_characters[b['character_idx'][s_idx].item()]
                    armature = mixamo_aramatures[character_name]
                    bs_rotations = recon_bs[s_idx].reshape(1, T, 5, 3, 4).clone()
                    armature.bs_rotations = bs_rotations
                    export_armature_animation(armature, output_dir / f'mano2mixamo{b_idx}_{s_idx}.mp4', animation=True, show_anchor=False, color='#FA7F6F')
                    # export_img(armature, '#FA7F6F', output_dir, f'mano2mixamo{b_idx}_{s_idx}')
                    mano_armature.reset_shape(shapes.permute(0, 2, 1)[s_idx].reshape(T, -1))
                    mano_armature.bs_rotations = b['bs_rotations'].permute(0, 4, 1, 2, 3)[s_idx].reshape(T, 1, 5, 3, 4)
                    export_armature_animation(mano_armature, output_dir / f'mano2mixamo{b_idx}_{s_idx}_gt.mp4', animation=True, show_anchor=False, color='white')
                    # export_img(mano_armature, 'white', output_dir, f'mano2mixamo{b_idx}_{s_idx}_gt')

            if 'offset' in b:
                recon_bs = cli.model.mano_forward(b, shapes, mano_m_tbs)
                if cli.model.pose_repr == 'ortho6d':
                    recon_bs = matrix_to_quaternion(rotation_6d_to_matrix(recon_bs))
                for s_idx in range(recon_bs.shape[0]):
                    if s_idx % 10 != 0:
                        continue
                    cli.model.mano_armature.reset_shape(shapes.permute(0, 2, 1)[s_idx].reshape(T, -1))
                    cli.model.mano_armature.bs_rotations = recon_bs[s_idx].reshape(T, 1, 5, 3, 4)
                    export_armature_animation(cli.model.mano_armature, Path(cli.config.output_dir) / f'mixamo2mano{b_idx}_{s_idx}.mp4', animation=True, show_anchor=False, color='#FA7F6F')
                    # export_img(cli.model.mano_armature, '#FA7F6F', output_dir, f'mixamo2mano{b_idx}_{s_idx}')
                    character_name = mixamo_characters[b['character_idx'][s_idx].item()]
                    armature = mixamo_aramatures[character_name]
                    armature.bs_rotations = b['bs_rotations'].permute(0, 4, 1, 2, 3)[s_idx].reshape(1, T, 5, 3, 4)
                    export_armature_animation(armature, Path(cli.config.output_dir) / f'mixamo2mano{b_idx}_{s_idx}_gt.mp4', animation=True, show_anchor=False, color='white')
                    # export_img(armature, 'white', output_dir, f'mixamo2mano{b_idx}_{s_idx}_gt')
