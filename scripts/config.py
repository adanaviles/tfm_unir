from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore
from omegaconf import OmegaConf


#############################
#  Define global variables  #
#############################
CURR_DIR = Path(__file__)
SRC_DIR = CURR_DIR.parents[1]
ROOT_DIR = CURR_DIR.parents[2]

##############################
#  Define config yaml paths  #
##############################
def get_config_path(relative_path):
    """
    Return the full path
    :param relative_path:
    :return:
    """
    return str(SRC_DIR / relative_path)

CONFIG_YAML_PATH_DATA = get_config_path("config/dailyaily_config/config.yaml")

##############################
#  Define abstract method    #
##############################
class AbstractSettings(ABC):
    """Abstract class of settings with general construction"""

    def __init__(self, config_path):
        self.ROOT_DIR = ROOT_DIR
        self.CURR_DIR = CURR_DIR
        self.SRC_DIR = SRC_DIR
        self.config: yaml.YAMLObject = yaml.safe_load(open(config_path))


@dataclass
class Config:
    k_folds: int


class Settings(AbstractSettings):
    """Meant to centralize the creation of settings and configuration"""

    def __init__(self):
        super().__init__(CONFIG_YAML_PATH_DATA)
        self.schema = OmegaConf.structured(Config)
        config_tester = OmegaConf.merge(self.schema, self.config)
        self.config = config_tester
