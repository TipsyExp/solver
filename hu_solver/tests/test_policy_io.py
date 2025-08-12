import unittest, tempfile
try:
    from hu_solver_native import SolverNative
except Exception:
    SolverNative = None

class TestPolicyIO(unittest.TestCase):
    def test_save_load_roundtrip(self):
        if SolverNative is None:
            self.skipTest("native module not available")
        with tempfile.TemporaryDirectory() as d:
            s1 = SolverNative(seed=42)
            s1.train(5, d)
            s1.save_policy(d)

            s2 = SolverNative(seed=999)
            s2.load_policy(d)

            state = {"street": 0, "to_call": 0.0, "last_raise": 0.0, "pos": 0}
            a = s1.query(state)
            b = s2.query(state)

            self.assertEqual(a, b)
            self.assertAlmostEqual(sum(a.values()), 1.0, places=9)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in a.values()))

if __name__ == "__main__":
    unittest.main()
