try:
    from hu_solver_native import SolverNative  # compiled module
except Exception:  # no native build yet
    SolverNative = None
