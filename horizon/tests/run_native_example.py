"""Run a native example with the extension selected by CTest."""
import importlib.util
import os
import runpy
import sys

import casadi  # Load CasADi before its C++ extension.

path = os.environ["HORIZON_PYILQR_EXTENSION"]
spec = importlib.util.spec_from_file_location("horizon.solvers.pyilqr", path)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)
sys.modules[spec.name] = extension
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
