from . import SolverNative as _Native
# ... existing code ...
class Solver:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.native = _Native(self.cfg.seed) if _Native else None
        # keep Python fallback if native is None

    def train(self, iters: int, out_dir: str):
        if self.native:
            self.native.train(iters, out_dir)
            self.native.save_policy(out_dir)
        else:
            # existing Python fallback path
            ...

    def load_policy(self, out_dir: str):
        if self.native:
            self.native.load_policy(out_dir)
        else:
            ...

    def query(self, state: dict):
        if self.native:
            return self.native.query(state)
        else:
            ...
