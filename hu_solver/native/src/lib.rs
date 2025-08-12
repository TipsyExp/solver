use pyo3::prelude::*;
use pyo3::types::{PyDict};
use rand::{rngs::StdRng, SeedableRng};

#[pyclass]
struct SolverNative {
    seed: u64,
    // TODO: add fields for regrets, averages, buckets, sizes, etc.
}

#[pymethods]
impl SolverNative {
    #[new]
    fn new(seed: u64) -> Self {
        SolverNative { seed }
    }

    /// Train ES-MCCFR + CFR+ (placeholder for now).
    fn train(&mut self, _iters: u64, _checkpoint_dir: &str) -> PyResult<()> {
        // TODO: implement real loop; keep deterministic via per-iter seeds derived from self.seed
        Ok(())
    }

    fn save_policy(&self, _checkpoint_dir: &str) -> PyResult<()> {
        // TODO: write policy.bin (v1, 32-byte header), index.bin (NodeId->offset),
        // meta.json with both SHA256 checksums; use your existing specs.
        Ok(())
    }

    fn load_policy(&mut self, _checkpoint_dir: &str) -> PyResult<()> {
        // TODO: read policy/index/meta exactly as spec'd
        Ok(())
    }

    /// Query(state_dict) -> {action: prob}
    fn query<'py>(&self, py: Python<'py>, state: &PyDict) -> PyResult<&'py PyDict> {
        // TODO: compute legal mask, map to BetStateId/NodeId, fetch probabilities
        // For now: deterministic dummy based on node_id hash
        let out = PyDict::new(py);
        out.set_item("fold", 0.17)?;
        out.set_item("call", 0.53)?;
        out.set_item("raise_to_0.33", 0.30)?;
        Ok(out)
    }
}

#[pymodule]
fn hu_solver_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SolverNative>()?;
    Ok(())
}
