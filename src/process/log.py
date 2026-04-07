# adapter: re-exports pipeline.log — imports the module file directly
# to avoid triggering pipeline/__init__.py (which loads preproc.py → cycle).
import importlib.util, os as _os
_mod = importlib.util.spec_from_file_location(
    "pipeline.log",
    _os.path.join(_os.path.dirname(__file__), "..", "pipeline", "log.py")
)
_m = importlib.util.module_from_spec(_mod)
_mod.loader.exec_module(_m)
Logger = _m.Logger
