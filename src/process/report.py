# adapter: re-exports pipeline.report directly from module file.
import importlib.util, os as _os
_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "pipeline", "report.py"))
_spec = importlib.util.spec_from_file_location("pipeline.report", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Reporter = _mod.Reporter
build_reports = _mod.build_reports
