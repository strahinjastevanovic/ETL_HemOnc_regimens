# adapter: surfaces Tracker from pipeline/audit.py under the new process.track home.
# When track.py is replaced with the real implementation, this shim disappears.
import importlib.util, os as _os, sys as _sys
_log_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "pipeline", "log.py"))
_log_spec = importlib.util.spec_from_file_location("pipeline.log", _log_path)
_log_mod = importlib.util.module_from_spec(_log_spec)
_sys.modules.setdefault("pipeline.log", _log_mod)
_log_spec.loader.exec_module(_log_mod)

_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "pipeline", "audit.py"))
_spec = importlib.util.spec_from_file_location("pipeline.audit", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Tracker = _mod.Tracker
