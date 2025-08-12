import os
import unittest
import yaml


class TestConfigSchema(unittest.TestCase):
    def test_config_has_seed_int(self):
        # looks for hu_solver/configs/hu_100bb.yaml (used in your workflow)
        cfg_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "hu_100bb.yaml")
        )
        if not os.path.isfile(cfg_path):
            self.skipTest(f"Config not found at {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # minimal shape check; expand later as you formalize the schema
        self.assertIn("seed", data, "config must define a 'seed'")
        self.assertIsInstance(data["seed"], int, "'seed' must be an integer")


if __name__ == "__main__":
    unittest.main()
