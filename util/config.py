##
# @package util
#

from datetime import datetime
from enum import Flag, auto
from os import path
from tomllib import load as loadtoml
from typing import Any, Dict, List, Optional

from util.logging import Logger, LogLevel
from util.patterns import Singleton


class ConfigPaths:
    artefacts: str = "artefacts"
    features: str
    base_feats: str

    freqs: str
    spmels: str
    monowavs: str

    trained_models: str
    models: str
    tensorboard: str
    latents: str
    logging: str

    raw_data: str
    raw_wavs: str
    proc_data: str

    raw_vctk: str
    raw_timit: str
    raw_uaspeech: str
    raw_smolspeech: str

    dataset_vctk: str
    dataset_timit: str
    dataset_uaspeech: str
    dataset_smolspeech: str


class ConfigModel:
    dim_con: int = 80
    dim_dec: int = 512
    dim_enc_1: int = 512
    dim_enc_2: int = 128
    dim_enc_3: int = 256
    dim_f0: int = 257
    dim_freq: int = 80
    dim_neck_1: int = 32
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
    file: Optional[str] = None


class RunTests(Flag):
    NOTHING = 0
    TRAIN = auto()
    SAVE_LATENTS = auto()
    SWAP_LATENTS = auto()
    SAVE_AUDIOS = auto()
    TEST = auto()


class ConfigOptions:
    bottleneck: str = "large"
    model_type: str = "SpeechSplit2"
    ntfy_url: str
    experiment: str
    dataset_name: str
    return_latents: bool = False
    trace: bool = False
    train: bool = False
    run_tests: RunTests = RunTests.NOTHING
    regenerate_data: bool = False
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


class ConfigDataLoader:
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 16
    samplier: int = 16


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
    dataloader: ConfigDataLoader = ConfigDataLoader()

    sample_rate: int = 16000
    batch_size: int = 1
    num_workers: int = 1

    ## Initialise configuration object
    # Reads the config toml file and creates a single object with the values
    def __init__(
        self,
        config_name: Optional[str] = None,
    ) -> None:
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if config_name is not None:
            if ".toml" not in config_name:
                self.original_config = f"configs/{config_name}.toml"
            else:
                self.original_config = config_name
            self.load_config(config_name)

    ## Load config file and update values
    #  Duplicate values will be overwritten, existing config options that not
    #  specified in the loaded files are not removed.
    def load_config(
        self,
        config_name: str,
    ) -> None:
        config_str = "configs/{}.toml"
        config_file = config_str.format(config_name)
        if not path.exists(config_file):
            Logger().fatal(f"Could not find file: {config_file}")
        tomldict = loadtoml(open(config_file, "rb"))
        if (
            "experiment" not in tomldict["options"].keys()
            and "bottleneck" not in tomldict["options"].keys()
        ):
            Logger().fatal(
                "Could not find options.experiment and "
                + "options.bottleneck in config file"
            )
        model_name = "{}-{}".format(
            tomldict["options"]["experiment"],
            tomldict["options"]["bottleneck"],
        )
        model_dict = loadtoml(open(config_str.format(model_name), "rb"))
        tomldict = self.__merge_dicts(tomldict, model_dict)
        # self.__print_config(tomldict)
        for key, subdict in tomldict.items():
            self.__map_categories(
                key,
                subdict,
            )
        self.__fill_nulls()

    def __print_config(self, config_dict: dict) -> None:
        Logger().info(
            "config:\n"
            + "\n".join(
                [
                    "\n".join(
                        [f"[{c}]"]
                        + [f"{k} = {v}" for k, v in config_dict[c].items()]
                        + [""]
                    )
                    for c in config_dict.keys()
                ]
            )
        )

    def __merge_dicts(self, dic1: dict, dic2: dict) -> dict:
        retval: dict = {}
        for cat in sorted({*dic1.keys(), *dic2.keys()}):
            retval[cat] = dict()
            if cat in sorted(dic1.keys()):
                for key, item in dic1[cat].items():
                    retval[cat][key] = item
            if cat in sorted(dic2.keys()):
                for key, item in dic2[cat].items():
                    retval[cat][key] = item
        return retval

    def __map_categories(
        self,
        key: str,
        subdict: Dict[str, Any],
    ) -> None:
        match key.lower():
            case "log":
                if "level" in subdict:
                    Logger().set_level(subdict["level"])
                if "file" in subdict:
                    if subdict["file"] is False:
                        self.__logging.file = None
                    elif subdict["file"] is True:
                        self.__logging.file = "SET_ME"
            case "paths":
                if not subdict.keys() >= {"raw_data", "proc_data"}:
                    Logger().fatal(
                        "Could not find paths.proc_data and paths.raw_data in config"
                    )
                self.paths.__dict__.update(subdict)
            case "model":
                self.model.__dict__.update(subdict)
            case "dataloader":
                self.dataloader.__dict__.update(subdict)
            case "training":
                self.training.__dict__.update(subdict)
            case "options":
                if not subdict.keys() >= {"dataset_name"}:
                    Logger().fatal("Could not find options.dataset_name in config")
                if "run_tests" in subdict.keys():
                    subdict["run_tests"] = self.__set_runtypes(subdict["run_tests"])
                self.options.__dict__.update(subdict)
                if not subdict.keys() >= {"experiment"}:
                    self.options.experiment = self.current_time
            case _:
                self.__dict__.update(subdict)
        return

    def __set_runtypes(self, runtype_list: List[str]) -> RunTests:
        runtype = RunTests.NOTHING
        for runtype_str in runtype_list:
            runtype_str = runtype_str.upper().strip()
            try:
                runtype |= RunTests[runtype_str]
            except Exception as e:
                Logger().fatal(
                    f"Invalid options.run_tests value in config: {runtype_str}\n"
                    f"Options: {[str(test) for test in RunTests]}\n"
                    f"Exception: {e}"
                )
        return runtype

    def __fill_nulls(self) -> None:
        self.__set_dataset_paths()
        self.__set_data_and_feat()
        self.__set_artefact_paths()
        if self.__logging.file == "SET_ME":
            self.__logging.file = f"{self.paths.logging}/{self.start_time}.log"
            Logger().set_file(self.__logging.file)

    def __set_artefact_paths(self) -> None:
        if not hasattr(self.paths, "logging"):
            self.paths.logging = f"{self.paths.artefacts}/logs"
        if not hasattr(self.paths, "trained_models"):
            self.paths.trained_models = f"{self.paths.artefacts}/trained_models"
        if not hasattr(self.paths, "tensorboard"):
            self.paths.tensorboard = f"{self.paths.artefacts}/tensorboard"
        if not hasattr(self.paths, "models"):
            self.paths.models = f"{self.paths.artefacts}/models"
        if not hasattr(self.paths, "latents"):
            self.paths.latents = f"{self.paths.artefacts}/latents"

        if not hasattr(self.paths, "freqs"):
            self.paths.freqs = f"{self.paths.features}/freqs"
        if not hasattr(self.paths, "spmels"):
            self.paths.spmels = f"{self.paths.features}/spmels"
        if not hasattr(self.paths, "monowavs"):
            self.paths.monowavs = f"{self.paths.features}/monowavs"

    def __set_dataset_paths(self) -> None:
        if not hasattr(self.paths, "raw_timit"):
            self.paths.raw_timit = f"{self.paths.raw_data}/TIMIT"
        if not hasattr(self.paths, "dataset_timit"):
            self.paths.dataset_timit = f"{self.paths.proc_data}/TIMIT"

        if not hasattr(self.paths, "raw_vctk"):
            self.paths.raw_vctk = f"{self.paths.raw_data}/VCTK-Corpus/wav"
        if not hasattr(self.paths, "dataset_vctk"):
            self.paths.dataset_vctk = f"{self.paths.proc_data}/VCTK-Corpus"

        if not hasattr(self.paths, "raw_smolspeech"):
            self.paths.raw_smolspeech = f"{self.paths.raw_data}/SmolSpeech"
        if not hasattr(self.paths, "dataset_smolspeech"):
            self.paths.dataset_smolspeech = f"{self.paths.proc_data}/SmolSpeech"

        if not hasattr(self.paths, "raw_uaspeech"):
            self.paths.raw_uaspeech = (
                f"{self.paths.raw_data}/UASpeech/audio/noisereduce"
            )
        if not hasattr(self.paths, "dataset_uaspeech"):
            self.paths.dataset_uaspeech = (
                f"{self.paths.proc_data}/UASpeech/audio/noisereduce"
            )

    def __set_data_and_feat(self) -> None:
        if self.options.dataset_name == "vctk":
            data_dir = self.paths.raw_vctk
            feat_dir = self.paths.dataset_vctk
        elif self.options.dataset_name == "smolspeech":
            data_dir = self.paths.raw_smolspeech
            feat_dir = self.paths.dataset_smolspeech
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
