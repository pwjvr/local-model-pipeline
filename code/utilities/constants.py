import os

from pathlib import Path
from config import Config

here_ = Path(os.path.dirname(os.path.abspath(__file__)))
config_file_ = here_ / 'config.yml'

conf = Config(config_file_)

SEED, VALIDATION_SIZE = conf.get_consts()

RAW_DTYPES = conf.get_dtypes()

KEPT_VARS = conf.all_variables

RAW_DF_PATH_PATTERN = "raw/df.pickle"
RAW_DF_SPLIT_PATH_PATTERN = "raw/df.cleaned.pickle"
RAW_DF_CLEANED_PATTERN = "raw/{dataset_type}.df.cleaned.pickle"
PREPROCESSOR_ARTEFACT_PATH_PATTERN = "processed/preprocessor.pickle"