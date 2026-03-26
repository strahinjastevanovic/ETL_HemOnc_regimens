# adapter: re-exports pipeline.handlers under the new singular name.
# Imports the module file directly to avoid pipeline/__init__ cycle.
import importlib.util, os as _os
_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "pipeline", "handlers.py"))
_spec = importlib.util.spec_from_file_location("pipeline.handlers", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Frame                = _mod.Frame
NullValueHandlers    = _mod.NullValueHandlers
RegimenHandler       = _mod.RegimenHandler
VariantHandler       = _mod.VariantHandler
PatternHandlers      = _mod.PatternHandlers
SupplementaryHandler = _mod.SupplementaryHandler
