##
# @package util
#

from datetime import datetime
from os import path
from tomllib import load as loadtoml
from typing import Any, Dict, Optional

from util.logging import Logger, LogLevel
from util.patterns import Singleton


class ConfigPaths:
    artefacts: str = "artefacts"
    features: str
    base_feats: str

    freqs: str
    spmels: str
    monowavs: str

    models: str
    tensorboard: str

    raw_data: str
    raw_wavs: str
    proc_data: str

    raw_vctk: str
    raw_timit: str
    raw_uaspeech: str

    dataset_vctk: str
    dataset_timit: str
    dataset_uaspeech: str


class ConfigModel:
    dim_con: int = 80
    dim_dec: int = 512
    dim_enc_1: int = 512
    dim_enc_2: int = 128
    dim_enc_3: int = 256
    dim_f0: int = 257
    dim_freq: int = 80
    dim_neck: int = 32
    dim_neck_2: int = 32
    dim_neck_3: int = 32
    dim_pit: int = 257
    dim_rhy: int = 80
    dim_spk_emb: int = 82

    chs_grp: int = 16
    dropout: float = 0.2
    freq_1: int = 1
    freq_2: int = 1
    freq_3: int = 1
    len_raw: int = 128
    max_len_pad: int = 192
    max_len_seg: int = 32
    max_len_seq: int = 128
    min_len_seg: int = 19
    min_len_seq: int = 64


class ConfigLogging:
    level: LogLevel = LogLevel.INFO
    path: str = "logs"
    file: Optional[str] = None


class ConfigOptions:
    bottleneck: str = "large"
    experiment: str
    dataset_name: str
    return_latents: bool = False
    trace: bool = False
    train: bool = True
    regenerate_data: bool = False
    regenerate_metadata: bool = False
    device_id: int = 0
    num_iters: int = 800000
    resume_iters: int = 0
    auto_resume: bool = False
    log_step: int = 100
    ckpt_save_step: int = 1000


class ConfigTraining:
    lr: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999


## Configuration object
# The Config object is a Singleton,
# The config file is generally only read once at the beginning of execution
class Config(metaclass=Singleton):
    current_time: str
    logfile: Optional[str] = None
    original_config: str

    __logging: ConfigLogging = ConfigLogging()
    paths: ConfigPaths = ConfigPaths()
    model: ConfigModel = ConfigModel()
    options: ConfigOptions = ConfigOptions()
    training: ConfigTraining = ConfigTraining()

    sample_rate: int = 16000
    batch_size: int = 1
    num_workers: int = 1

    ## Initialise configuration object
    # Reads the config toml file and creates a single object with the values
    def __init__(
        self,
        config_file: str,
    ) -> None:
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.original_config = config_file
        self.load_config(config_file)

    ## Load config file and update values
    #  Duplicate values will be overwritten, existing config options that not
    #  specified in the loaded files are not removed.
    def load_config(
        self,
        config_file: str,
    ) -> None:
        if not path.exists(config_file):
            Logger().fatal(f"Could not find file: {config_file}")
        for key, subdict in loadtoml(
            open(config_file, "rb"),
        ).items():
            self.__map_categories(
                key,
                subdict,
            )
        self.__fill_nulls()

    def __map_categories(
        self,
        key: str,
        subdict: Dict[str, Any],
    ) -> None:
        match key.lower():
            case "log":
                if "level" in subdict:
                    Logger().set_level_str(subdict["level"])
                if "path" in subdict:
                    self.__logging.path = subdict["path"]
                if "file" in subdict:
                    if subdict["file"] is False:
                        self.__logging.file = None
                    elif subdict["file"] is True:
                        self.__logging.file = (
                            f"{self.__logging.path}/{self.start_time}.log"
                        )
                    elif "/" in subdict["file"]:
                        self.__logging.file = subdict["file"]
                    else:
                        self.__logging.file = f"{self.__logging.path}/{subdict["file"]}"
                if self.__logging.file is not None:
                    Logger().set_file(self.__logging.file)
            case "paths":
                if not subdict.keys() >= {"raw_data", "proc_data"}:
                    Logger().fatal(
                        "Could not find paths.proc_data and paths.raw_data in config"
                    )
                self.paths.__dict__.update(subdict)
            case "model":
                self.model.__dict__.update(subdict)
            case "training":
                self.training.__dict__.update(subdict)
            case "options":
                if not subdict.keys() >= {"dataset_name"}:
                    Logger().fatal("Could not find options.dataset_name in config")
                self.options.__dict__.update(subdict)
                if not subdict.keys() >= {"experiment"}:
                    self.options.experiment = self.current_time
                if self.options.regenerate_data:
                    self.options.regenerate_metadata = True
            case _:
                self.__dict__.update(subdict)
        return

    def __fill_nulls(
        self,
    ) -> None:

        if not hasattr(self.paths, "raw_timit"):
            self.paths.raw_timit = f"{self.paths.raw_data}/TIMIT"
        if not hasattr(self.paths, "dataset_timit"):
            self.paths.dataset_timit = f"{self.paths.proc_data}/TIMIT"
        if not hasattr(self.paths, "raw_vctk"):
            self.paths.raw_vctk = f"{self.paths.raw_data}/VCTK-Corpus/wav"
        if not hasattr(self.paths, "dataset_vctk"):
            self.paths.dataset_vctk = f"{self.paths.proc_data}/VCTK-Corpus"
        if not hasattr(self.paths, "raw_uaspeech"):
            self.paths.raw_uaspeech = (
                f"{self.paths.raw_data}/UASpeech/audio/noisereduce"
            )
        if not hasattr(self.paths, "dataset_uaspeech"):
            self.paths.dataset_uaspeech = (
                f"{self.paths.proc_data}/UASpeech/audio/noisereduce"
            )

        self.__set_data_and_feat()

        if not hasattr(self.paths, "tensorboard"):
            self.paths.tensorboard = f"{self.paths.artefacts}/tensorboard"
        if not hasattr(self.paths, "models"):
            self.paths.models = f"{self.paths.artefacts}/models"
        if not hasattr(self.paths, "freqs"):
            self.paths.freqs = f"{self.paths.features}/freqs"
        if not hasattr(self.paths, "spmels"):
            self.paths.spmels = f"{self.paths.features}/spmels"
        if not hasattr(self.paths, "monowavs"):
            self.paths.monowavs = f"{self.paths.features}/monowavs"

    def __set_data_and_feat(
        self,
    ) -> None:
        if self.options.dataset_name == "vctk":
            data_dir = self.paths.raw_vctk
            feat_dir = self.paths.dataset_vctk
        elif self.options.dataset_name == "uaspeech":
            data_dir = self.paths.raw_uaspeech
            feat_dir = self.paths.dataset_uaspeech
        elif self.options.dataset_name == "timit":
            data_dir = self.paths.raw_timit
            feat_dir = self.paths.dataset_timit
        else:
            Logger().fatal(
                "Invalid options.dataset_name in config: {}".format(
                    self.options.dataset_name
                )
            )
        self.paths.features = feat_dir
        self.paths.raw_wavs = data_dir
        return
