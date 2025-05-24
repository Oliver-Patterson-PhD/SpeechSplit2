__all__ = [
    "Config",
    "RunTests",
]

from datetime import datetime
from enum import Flag, auto
from tomllib import load as loadtoml
from typing import Any, Dict, List, Optional, Self

from .file import exists, path
from .logging import Logger, LogLevel
from .patterns import Singleton


class ConfigPaths:
    artefacts: str = "artefacts"
    features: str
    base_feats: str

    freqs: str
    spmels: str
    monowavs: str
    fullwavs: str
    phases: str
    cleanwavs: str

    full_models: str
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
    raw_smolvctk: str

    dataset_vctk: str
    dataset_timit: str
    dataset_uaspeech: str
    dataset_smolspeech: str
    dataset_smolvctk: str


class ConfigAudioProcessing:
    chunk_length: int = 30
    f0_f_hi: int = 600
    f0_f_lo: int = 100
    f0_m_hi: int = 250
    f0_m_lo: int = 50
    freq_max: int = 7600
    freq_min: int = 90
    hi_pass_cutoff: int = 30
    hop_len: int = 160
    max_len_pad = 192
    n_fft: int = 400
    sample_rate: int = 16000
    vtlp_fft: int = 400 * 2
    fold_div: int = 2


class ConfigModel:
    freq_1: int
    freq_2: int
    freq_3: int
    dim_neck_1: int
    dim_neck_2: int
    dim_neck_3: int  # 32

    dim_con: int = 80  # N_MELS
    dim_dec: int = 512  # HOP_LENGTH * 2
    dim_enc_1: int = 512  # HOP_LENGTH * 2
    dim_enc_2: int = 128  # HOP_LENGTH / 2
    dim_enc_3: int = 256  # HOP_LENGTH
    dim_f0: int = 257  # HOP_LENGTH + 1
    dim_freq: int = 80  # N_MELS
    dim_pit: int = 257  # HOP_LENGTH + 1
    dim_rhy: int = 80  # N_MELS
    dim_spk_emb: int = 82  # N_MELS + 2

    chs_grp: int = 16
    dropout: float = 0.2
    len_raw: int = 128  # HOP_LENGTH / 2
    max_len_pad: int = 192
    max_len_seg: int = 32
    max_len_seq: int = 128  # HOP_LENGTH / 2
    min_len_seg: int = 19
    min_len_seq: int = 64


class ConfigLogging:
    level: LogLevel = LogLevel.INFO
    file: Optional[str] = None
    callgraph: bool = False


class RunTests(Flag):
    NOTHING = 0
    TRAIN = auto()
    SYLLABLE_ESTIMATION = auto()
    SAVE_LATENTS = auto()
    SWAP_LATENTS = auto()
    SAVE_AUDIOS = auto()
    TEST = auto()


class ConfigOptions:
    bottleneck: str = "large"
    model_type: str = "SpeechSplit2"
    whisper_type: str = "large-v3-turbo"
    parallelwavegan_name: str = "parallelwavegan-3M"
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
    mask_loss: bool = False
    lr: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999


class ConfigDataLoader:
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 16
    samplier: int = 16
    drop_and_pad: bool = True


## Configuration object
# The Config object is a Singleton,
# The config file is generally only read once at the beginning of execution
class Config(metaclass=Singleton):
    start_time: str
    logfile: Optional[str] = None
    original_config: str

    __logging: ConfigLogging = ConfigLogging()
    audio: ConfigAudioProcessing = ConfigAudioProcessing()
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
    def __init__(self: Self, config_name: Optional[str] = None) -> None:
        self.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if config_name is not None:
            if ".toml" not in config_name:
                self.original_config = f"configs/{config_name}.toml"
            else:
                self.original_config = config_name
            self.load_config(config_name)
        else:
            raise RuntimeError("Config not initialised")

    ## Load config file and update values
    #  Duplicate values will be overwritten, existing config options that not
    #  specified in the loaded files are not removed.
    def load_config(self: Self, config_name: str) -> None:
        config_str = "configs/{}.toml"
        config_file = config_str.format(config_name)
        if not exists(config_file):
            err_str = f"Could not find file: {config_file}"
            Logger().fatal(err_str)
            raise FileNotFoundError(err_str)
        tomldict = loadtoml(open(config_file, "rb"))
        if (
            "experiment" not in tomldict["options"].keys()
            and "bottleneck" not in tomldict["options"].keys()
        ):
            err_str = "Could not find options.experiment and options.bottleneck in config file"
            Logger().fatal(err_str)
            raise RuntimeError(err_str)
        elif config_name == "scratch":
            model_name = "models/" + tomldict["options"]["bottleneck"]
            model_dict = loadtoml(open(config_str.format(model_name), "rb"))
            tomldict = self.__merge_dicts(tomldict, model_dict, {})
        else:
            model_name = "models/" + tomldict["options"]["bottleneck"]
            model_dict = loadtoml(open(config_str.format(model_name), "rb"))
            experiment_name = "experiments/" + tomldict["options"]["experiment"]
            experiment_dict = loadtoml(open(config_str.format(experiment_name), "rb"))
            audio_name = "audio/" + tomldict["options"]["audio"]
            audio_dict = loadtoml(open(config_str.format(audio_name), "rb"))
            tomldict = self.__merge_dicts(
                tomldict,
                model_dict,
                experiment_dict,
                audio_dict,
            )
        self.__print_config(tomldict)
        for key, subdict in tomldict.items():
            self.__map_categories(
                key,
                subdict,
            )
        self.__fill_nulls()

    def __print_config(self: Self, config_dict: dict) -> None:
        Logger().info(
            f"config: {self.original_config}\n"
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

    def __merge_dicts(self: Self, *dict_args) -> dict:
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def __map_categories(self: Self, key: str, subdict: Dict[str, Any]) -> None:
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
                    err_str = (
                        "Could not find paths.proc_data and paths.raw_data in config"
                    )
                    Logger().fatal(err_str)
                    raise RuntimeError(err_str)
                self.paths.__dict__.update(subdict)
            case "model":
                self.model.__dict__.update(subdict)
            case "dataloader":
                self.dataloader.__dict__.update(subdict)
            case "audio":
                self.audio.__dict__.update(subdict)
            case "training":
                self.training.__dict__.update(subdict)
            case "options":
                if not subdict.keys() >= {"dataset_name"}:
                    err_str = "Could not find options.dataset_name in config"
                    Logger().fatal(err_str)
                    raise RuntimeError(err_str)
                if "run_tests" in subdict.keys():
                    subdict["run_tests"] = self.__set_runtypes(subdict["run_tests"])
                self.options.__dict__.update(subdict)
                if not subdict.keys() >= {"experiment"}:
                    self.options.experiment = self.start_time
            case _:
                self.__dict__.update(subdict)
        return

    def __set_runtypes(self: Self, runtype_list: List[str]) -> RunTests:
        runtype = RunTests.NOTHING
        for runtype_str in runtype_list:
            runtype_str = runtype_str.upper().strip()
            try:
                runtype |= RunTests[runtype_str]
            except Exception as e:
                err_str = (
                    f"Invalid options.run_tests value in config: {runtype_str}\n"
                    f"Options: {[str(test) for test in RunTests]}"
                )
                Logger().fatal(err_str)
                raise Exception(err_str) from e
        return runtype

    def __fill_nulls(self: Self) -> None:
        self.__set_dataset_paths()
        self.__set_data_and_feat()
        self.__set_artefact_paths()
        if self.__logging.file == "SET_ME":
            self.__logging.file = path(
                self.paths.logging,
                f"{self.start_time}-{self.options.experiment}.log",
            )
            Logger().set_file(self.__logging.file)
        if self.__logging.callgraph:
            Logger().enable_callgraph()

    def __set_artefact_paths(self: Self) -> None:
        if not hasattr(self.paths, "logging"):
            self.paths.logging = path(self.paths.artefacts, "logs")
        if not hasattr(self.paths, "full_models"):
            self.paths.full_models = path(self.paths.artefacts, "full_models")
        if not hasattr(self.paths, "tensorboard"):
            self.paths.tensorboard = path(self.paths.artefacts, "tensorboard")
        if not hasattr(self.paths, "models"):
            self.paths.models = path(self.paths.artefacts, "models")
        if not hasattr(self.paths, "latents"):
            self.paths.latents = path(self.paths.artefacts, "latents")

        if not hasattr(self.paths, "freqs"):
            self.paths.freqs = path(self.paths.features, "freqs")
        if not hasattr(self.paths, "spmels"):
            self.paths.spmels = path(self.paths.features, "spmels")
        if not hasattr(self.paths, "monowavs"):
            self.paths.monowavs = path(self.paths.features, "monowavs")
        if not hasattr(self.paths, "fullwavs"):
            self.paths.fullwavs = path(self.paths.features, "fullwavs")
        if not hasattr(self.paths, "phases"):
            self.paths.phases = path(self.paths.features, "phases")
        if not hasattr(self.paths, "cleanwavs"):
            self.paths.cleanwavs = path(self.paths.features, "cleanwavs")

    def __set_dataset_paths(self: Self) -> None:
        if not hasattr(self.paths, "raw_timit"):
            self.paths.raw_timit = path(self.paths.raw_data, "TIMIT")
        if not hasattr(self.paths, "dataset_timit"):
            self.paths.dataset_timit = path(self.paths.proc_data, "TIMIT")

        if not hasattr(self.paths, "raw_vctk"):
            self.paths.raw_vctk = path(self.paths.raw_data, "VCTK-Corpus", "wav")
        if not hasattr(self.paths, "dataset_vctk"):
            self.paths.dataset_vctk = path(self.paths.proc_data, "VCTK-Corpus")

        if not hasattr(self.paths, "raw_uaspeech"):
            self.paths.raw_uaspeech = path(self.paths.raw_data, "UASpeech", "audio", "original")
        if not hasattr(self.paths, "dataset_uaspeech"):
            self.paths.dataset_uaspeech = (
                path(self.paths.proc_data, "UASpeech", "audio", "original")
            )

        if not hasattr(self.paths, "raw_smolspeech"):
            self.paths.raw_smolspeech = path(self.paths.raw_data, "SmolSpeech")
        if not hasattr(self.paths, "dataset_smolspeech"):
            self.paths.dataset_smolspeech = path(self.paths.proc_data, "SmolSpeech")

        if not hasattr(self.paths, "raw_smolvctk"):
            self.paths.raw_smolvctk = path(self.paths.raw_data, "SmolVCTK")
        if not hasattr(self.paths, "dataset_smolvctk"):
            self.paths.dataset_smolvctk = path(self.paths.proc_data, "SmolVCTK")

    def __set_data_and_feat(self: Self) -> None:
        if self.options.dataset_name == "vctk":
            data_dir = self.paths.raw_vctk
            feat_dir = self.paths.dataset_vctk
        elif self.options.dataset_name == "uaspeech":
            data_dir = self.paths.raw_uaspeech
            feat_dir = self.paths.dataset_uaspeech
        elif self.options.dataset_name == "timit":
            data_dir = self.paths.raw_timit
            feat_dir = self.paths.dataset_timit
        elif self.options.dataset_name == "smolspeech":
            data_dir = self.paths.raw_smolspeech
            feat_dir = self.paths.dataset_smolspeech
        elif self.options.dataset_name == "smolvctk":
            data_dir = self.paths.raw_smolvctk
            feat_dir = self.paths.dataset_smolvctk
        else:
            err_str = (
                f"Invalid options.dataset_name in config: {self.options.dataset_name}"
            )
            Logger().fatal(err_str)
            raise RuntimeError(err_str)
        self.paths.features = feat_dir
        self.paths.raw_wavs = data_dir
