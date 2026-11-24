import os as _os

from . import paths as _paths

_os.environ["HF_HOME"] = str(_paths.hf_home())
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
