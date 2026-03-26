# adapter: re-exports pipeline.resolvers under the new name.
# Also provides forward-compatible aliases for the four resolver classes
# that will be introduced in Phase 2. Until then they all point at Resolver.
import importlib.util, os as _os
_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "pipeline", "resolvers.py"))
_spec = importlib.util.spec_from_file_location("pipeline.resolvers", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Resolver = _mod.Resolver

# Forward aliases — replaced by real classes in Phase 2
ResolverParted     = Resolver   # ACTION 3.1
ResolverIndefinite = Resolver   # ACTION 3.2
ResolverAllDays    = Resolver   # ACTION 3.3

class ResolverKey:              # ACTION 3.4 — stub, pass-through
    @staticmethod
    def with_variant_key(frame):
        """Stub: returns frame unchanged until Phase 3 (variant_key introduction)."""
        return frame
