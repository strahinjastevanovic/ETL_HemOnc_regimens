import os
import importlib.util

# Load `preprocessing` from sibling `preproc.py` (the file, not the package)
mod_path = os.path.join(os.path.dirname(__file__), "..", "process.py")
mod_path = os.path.abspath(mod_path)

spec = importlib.util.spec_from_file_location("preproc_runner", mod_path)
preproc_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preproc_runner)

# Export from this package
preprocessing = preproc_runner.preprocessing

from .audit import *
from .controller import Preprocessor
from .handle import *
from .log import *
from .report import *
from .resolve import *
