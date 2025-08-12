import unittest

try:
    from hu_solver_native import SolverNative
except Exception:
    SolverNative = None


class TestDeterminism(unittest.TestCase):
    def test_query_is_deterministic(self):
        if SolverNative is None:
            self.skipTest("native module not available")
        s = SolverNative(seed=1234)
        state = {"any": "thing"}  # native currently ignores it
        a = s.query(state)
        b = s.query(state)

        # same output each time
        self.assertEqual(a, b)

        # probs form a distribution
        self.assertAlmostEqual(sum(a.values()), 1.0, places=9)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in a.values()))

    def test_train_save_load_no_crash(self):
        if SolverNative is None:
            self.skipTest("native module not available")
        s = SolverNative(seed=1)
        # native methods are placeholders, just ensure they don't throw
        s.train(10, "policies/tmp")
        s.save_policy("policies/tmp")
        s.load_policy("policies/tmp")


if __name__ == "__main__":
    unittest.main()
