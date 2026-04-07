import os
import importlib.util

mod_path = os.path.join(os.path.dirname(__file__), "..", "query_vocab.py")
mod_path = os.path.abspath(mod_path)

spec = importlib.util.spec_from_file_location("query_runner", mod_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from .vocab import query_valid_drugs, query_conditions