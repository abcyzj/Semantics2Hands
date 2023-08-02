from pytorch_lightning.cli import LightningCLI

from data.combined_motion import MixedDataModule
from model.tbs import TBSNet
from utils.pl import NoCKPTSaveConfigCallback
from utils.armature_config import config_mixamo_armature


class TBSCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(nested_key='optim', link_to='model.optim_init')


def cli_main():
    config_mixamo_armature()
    TBSCLI(TBSNet, MixedDataModule, save_config_callback=NoCKPTSaveConfigCallback)


if __name__ == '__main__':
    cli_main()
