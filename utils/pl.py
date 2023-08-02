from jsonargparse import Namespace
from pytorch_lightning.cli import LightningArgumentParser, SaveConfigCallback


class NoCKPTSaveConfigCallback(SaveConfigCallback):
    def __init__(self, parser: LightningArgumentParser, config: Namespace, config_filename: str, overwrite: bool = False, multifile: bool = False) -> None:
        config.pop('ckpt_path') # do not save ckpt_path in the config file
        super().__init__(parser, config, config_filename, overwrite, multifile)
