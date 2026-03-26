# adapter: surfaces Preprocessor from pipeline/main.py under the new module path.
# All pipeline.* sub-dependencies are pre-loaded into sys.modules directly from
# their module files so that pipeline/__init__.py (which loads preproc.py) is
# never triggered during this import chain.
import importlib.util, os as _os, sys as _sys

def _load(name, rel):
    """Load a pipeline sub-module directly by file path and register in sys.modules."""
    path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", rel))
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    _sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod

_load("pipeline.log",       "pipeline/log.py")
_load("pipeline.audit",     "pipeline/audit.py")
_load("pipeline.report",    "pipeline/report.py")
_load("pipeline.handlers",  "pipeline/handlers.py")
_load("pipeline.resolvers", "pipeline/resolvers.py")
_main = _load("pipeline.main",  "pipeline/main.py")

Preprocessor = _main.Preprocessor

