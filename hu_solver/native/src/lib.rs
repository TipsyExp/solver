use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::BTreeMap;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};
use sha2::{Sha256, Digest};
use serde_json::json;

/// Quantize a big‑blind amount into Q4 fixed point (scale 16) using round‑half‑to‑even.
/// Caps values at 255.9375 BB as specified in the Phase 1 spec.
fn quantize_bb(bb: f64) -> u32 {
    // Cap the value
    let mut val = if bb.is_finite() { bb } else { 0.0 };
    if val < 0.0 { val = 0.0; }
    if val > 255.9375 { val = 255.9375; }
    // Scale
    let scaled = val * 16.0;
    let floor = scaled.floor();
    let diff = scaled - floor;
    let mut rounded = if diff > 0.5 {
        floor + 1.0
    } else if diff < 0.5 {
        floor
    } else {
        // exactly .5 — round to even
        if ((floor as u64) % 2) == 0 { floor } else { floor + 1.0 }
    };
    if rounded < 0.0 { rounded = 0.0; }
    if rounded > 4095.0 { rounded = 4095.0; }
    rounded as u32
}

/// Build a BetStateId (32 bits) from quantized to_call/last_raise, jam flag, position and street.
fn build_bet_state_id(to_call_bb: f64, last_raise_bb: f64, jam_flag: bool, pos: u8, street: u8) -> u32 {
    let to_call_q = quantize_bb(to_call_bb);
    let last_raise_q = quantize_bb(last_raise_bb);
    let jam = if jam_flag { 1u32 } else { 0u32 };
    let pos_bit = (pos as u32) & 0x1;
    let street_bits = (street as u32) & 0x3;
    // Bit layout: to_call_q[11:0], last_raise_q[23:12], jam[24], pos[25], street[27:26], reserved[31:28]
    (to_call_q & 0xFFF)
        | ((last_raise_q & 0xFFF) << 12)
        | (jam << 24)
        | (pos_bit << 25)
        | (street_bits << 26)
}

/// Build a NodeId (64 bits) from board_bucket, hand_bucket, bet_state_id and street.
fn build_node_id(board_bucket: u16, hand_bucket: u16, bet_state_id: u32, street: u8) -> u64 {
    let bb = (board_bucket as u64) & 0x7FF; // 11 bits
    let hb = (hand_bucket as u64) & 0x7FF; // 11 bits
    let bs = bet_state_id as u64 & 0xFFFF_FFFF;
    let st = (street as u64) & 0x3;
    (bb << 53) | (hb << 42) | (bs << 10) | (st << 8)
}

/// A simple deterministic hash used to derive per‑iteration seeds from a global seed.
fn derive_iter_seed(base_seed: u64, iter: u64) -> u64 {
    // SplitMix64 style hash: https://en.wikipedia.org/wiki/Splitmix64
    let mut z = base_seed.wrapping_add(iter.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[pyclass]
#[derive(Clone)]
pub struct SolverNative {
    seed: u64,
    num_actions: usize,
    action_keys: Vec<String>,
    node_map: BTreeMap<u64, usize>,
    avg_strategy: Vec<Vec<f64>>, // accumulate before normalising
    iterations: u64,
}

#[pymethods]
impl SolverNative {
    #[new]
    fn new(seed: u64) -> Self {
        // Define the fixed action set: fold, call, raise_to_0.33, raise_to_0.75, raise_to_1.25, jam
        let action_keys = vec![
            "fold".to_string(),
            "call".to_string(),
            "raise_to_0.33".to_string(),
            "raise_to_0.75".to_string(),
            "raise_to_1.25".to_string(),
            "jam".to_string(),
        ];
        let num_actions = action_keys.len();
        SolverNative {
            seed,
            num_actions,
            action_keys,
            node_map: BTreeMap::new(),
            avg_strategy: Vec::new(),
            iterations: 0,
        }
    }

    /// Train a very simple, deterministic ES‑MCCFR loop.  This implementation does not
    /// enumerate the full NLHE tree—it merely generates deterministic pseudo‑random
    /// strategy values for two canonical NodeIds (0 and 1) to illustrate the on‑disk
    /// format and deterministic seeding requirements.  Replace this logic with a
    /// full MCCFR traversal when extending the solver.
    fn train(&mut self, iters: u64, _checkpoint_dir: &str) -> PyResult<()> {
        // Ensure that the two example NodeIds exist and have storage allocated
        for node_id in [0u64, 1u64].iter() {
            self.node_map.entry(*node_id).or_insert_with(|| {
                let idx = self.avg_strategy.len();
                self.avg_strategy.push(vec![0.0f64; self.num_actions]);
                idx
            });
        }
        // For each iteration, derive a deterministic RNG and accumulate random values
        for i in 0..iters {
            let iter_seed = derive_iter_seed(self.seed, self.iterations + i);
            let mut rng = StdRng::seed_from_u64(iter_seed);
            for (_node_id, &idx) in self.node_map.iter() {
                // Generate pseudo‑random positive weights
                let mut vals: Vec<f64> = (0..self.num_actions)
                    .map(|_| rng.gen_range(0.0f64..1.0))
                    .collect();
                // Normalise to avoid extremely small numbers
                let sum: f64 = vals.iter().sum();
                if sum > 0.0 {
                    for v in vals.iter_mut() {
                        *v /= sum;
                    }
                }
                // Accumulate into avg_strategy
                for a in 0..self.num_actions {
                    self.avg_strategy[idx][a] += vals[a];
                }
            }
        }
        self.iterations += iters;
        Ok(())
    }

    /// Save the average strategy to policy.bin, index.bin and meta.json in `checkpoint_dir`.
    fn save_policy(&self, checkpoint_dir: &str) -> PyResult<()> {
        // Create directory if needed
        create_dir_all(checkpoint_dir).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        // Prepare policy header
        let actions_per_node = self.num_actions as u16;
        let nodes = self.avg_strategy.len() as u64;
        let mut header = [0u8; 32];
        // magic 'PPOL'
        header[0..4].copy_from_slice(&0x50504F4Cu32.to_le_bytes());
        header[4..6].copy_from_slice(&1u16.to_le_bytes());
        header[6] = 1; // little endian
        header[7] = 0; // fixed actions
        header[8] = 1; // dtype = float32
        header[9] = 0; // reserved
        header[10..12].copy_from_slice(&actions_per_node.to_le_bytes());
        header[12..14].copy_from_slice(&32u16.to_le_bytes());
        header[14..22].copy_from_slice(&nodes.to_le_bytes());
        header[22..26].copy_from_slice(&0u32.to_le_bytes()); // body_checksum unused
        header[26..30].copy_from_slice(&0u32.to_le_bytes()); // reserved
        // Write policy.bin
        let policy_path = format!("{}/policy.bin", checkpoint_dir);
        let mut policy_file = File::create(&policy_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        policy_file.write_all(&header).unwrap();
        // Flatten and normalise the average strategies to probabilities
        for row in &self.avg_strategy {
            let sum: f64 = row.iter().sum();
            let normaliser = if sum > 0.0 { sum } else { 1.0 };
            for v in row {
                let p: f32 = (*v / normaliser) as f32;
                policy_file.write_all(&p.to_le_bytes()).unwrap();
            }
        }
        policy_file.flush().unwrap();
        // Compute policy.bin SHA256
        let policy_hash_hex = {
            let mut hasher = Sha256::new();
            let mut f = File::open(&policy_path).unwrap();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            hasher.update(&buf);
            let result = hasher.finalize();
            hex::encode(result)
        };
        // Build index.bin
        let index_path = format!("{}/index.bin", checkpoint_dir);
        let mut index_file = File::create(&index_path).unwrap();
        let mut offset = 0u64;
        for (&node_id, _) in &self.node_map {
            index_file.write_all(&node_id.to_le_bytes()).unwrap();
            index_file.write_all(&offset.to_le_bytes()).unwrap();
            offset += (self.num_actions as u64) * 4; // each f32 is 4 bytes
        }
        index_file.flush().unwrap();
        // Compute index.bin SHA256
        let index_hash_hex = {
            let mut hasher = Sha256::new();
            let mut f = File::open(&index_path).unwrap();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            hasher.update(&buf);
            let result = hasher.finalize();
            hex::encode(result)
        };
        // Compose meta.json
        let meta = json!({
            "schema_version": 1,
            "policy_header": {"version": 1, "dtype": "f32", "actions_mode": "fixed", "actions_per_node": actions_per_node},
            "index": {"format": "node_id", "entry": "fixed", "sorted": true},
            "bucket_counts": {"board": 64, "hand": 192},
            "bet_state_id": {"version": 1, "scale_bb_q": 16, "caps": {"to_call_bb": 255.9375, "last_raise_bb": 255.9375}},
            "node_id_bits": {"board_bucket": 11, "hand_bucket": 11, "bet_state_id": 32, "street": 2},
            "discrete_sizes": {"flop": [0.33, 0.75, 1.25], "turn": [0.75, 1.25], "river": [1.0, "jam"]},
            "checksum": {"policy_bin_sha256": policy_hash_hex, "index_bin_sha256": index_hash_hex},
            "total_nodes": nodes,
            "policy_header_echo": {"version": 1, "dtype": "f32", "actions_mode": "fixed", "actions_per_node": actions_per_node}
        });
        let meta_path = format!("{}/meta.json", checkpoint_dir);
        let mut meta_file = File::create(&meta_path).unwrap();
        meta_file.write_all(meta.to_string().as_bytes()).unwrap();
        meta_file.flush().unwrap();
        Ok(())
    }

    /// Load a previously saved policy into memory.  This resets the internal
    /// strategy table and node map to match the loaded files.
    fn load_policy(&mut self, checkpoint_dir: &str) -> PyResult<()> {
        // Read policy.bin
        let policy_path = format!("{}/policy.bin", checkpoint_dir);
        let mut policy_file = File::open(&policy_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut header = [0u8; 32];
        policy_file.read_exact(&mut header).unwrap();
        let actions_per_node = u16::from_le_bytes([header[10], header[11]]) as usize;
        let nodes = u64::from_le_bytes(header[14..22].try_into().unwrap()) as usize;
        // Reinitialise structures
        self.num_actions = actions_per_node;
        self.avg_strategy = vec![vec![0.0f64; actions_per_node]; nodes];
        // Read body as f32 and convert to f64
        let mut body = Vec::new();
        policy_file.read_to_end(&mut body).unwrap();
        let mut offset = 0;
        for i in 0..nodes {
            for a in 0..actions_per_node {
                let bytes: [u8; 4] = body[offset..offset+4].try_into().unwrap();
                let val = f32::from_le_bytes(bytes) as f64;
                self.avg_strategy[i][a] = val;
                offset += 4;
            }
        }
        // Read index.bin
        let index_path = format!("{}/index.bin", checkpoint_dir);
        let mut index_file = File::open(&index_path).unwrap();
        let mut index_bytes = Vec::new();
        index_file.read_to_end(&mut index_bytes).unwrap();
        self.node_map.clear();
        let entry_size = 16;
        for i in 0..(index_bytes.len() / entry_size) {
            let base = i * entry_size;
            let node_id = u64::from_le_bytes(index_bytes[base..base+8].try_into().unwrap());
            let offset_bytes = u64::from_le_bytes(index_bytes[base+8..base+16].try_into().unwrap());
            let row = (offset_bytes / (self.num_actions as u64 * 4)) as usize;
            self.node_map.insert(node_id, row);
        }
        Ok(())
    }

    /// Query the policy for a given state.  The input is a Python dict with keys:
    /// {street, to_call, last_raise, pos}.  Board and hand buckets are stubbed
    /// (always 0) because full abstractions are not yet integrated.  The method
    /// returns a dict mapping action names to probabilities.  Unknown NodeIds
    /// return a uniform distribution over actions.
fn query<'py>(&self, py: Python<'py>, state: &PyDict) -> PyResult<&'py PyDict> {
    // street: accept string ("preflop"/"flop"/"turn"/"river"), or int 0..3, or default to 0 (preflop)
    let street: u8 = match state.get_item("street")? {
        Some(obj) => {
            if let Ok(s) = obj.extract::<&str>() {
                match s {
                    "preflop" => 0,
                    "flop" => 1,
                    "turn" => 2,
                    "river" => 3,
                    _ => 0,
                }
            } else if let Ok(i) = obj.extract::<u8>() {
                if i <= 3 { i } else { 0 }
            } else {
                0
            }
        }
        None => 0,
    };

    // Optional fields with defaults
    let to_call: f64 = match state.get_item("to_call")? {
        Some(v) => v.extract().unwrap_or(0.0),
        None => 0.0,
    };
    let last_raise: f64 = match state.get_item("last_raise")? {
        Some(v) => v.extract().unwrap_or(0.0),
        None => 0.0,
    };
    let pos: u8 = match state.get_item("pos")? {
        Some(v) => v.extract().unwrap_or(0u8),
        None => 0u8,
    };

    let bet_state_id = build_bet_state_id(to_call, last_raise, true, pos, street);
    let node_id = build_node_id(0, 0, bet_state_id, street);

    let out = PyDict::new(py);
    if let Some(&row) = self.node_map.get(&node_id) {
        let probs = &self.avg_strategy[row];
        let sum: f64 = probs.iter().sum();
        let normaliser = if sum > 0.0 { sum } else { self.num_actions as f64 };
        for (i, key) in self.action_keys.iter().enumerate() {
            let p = probs[i] / normaliser;
            out.set_item(key, p as f64)?;
        }
    } else {
        let uniform = 1.0f64 / (self.num_actions as f64);
        for key in &self.action_keys {
            out.set_item(key, uniform)?;
        }
    }
    Ok(out)
}

}

#[pymodule]
fn hu_solver_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SolverNative>()?;
    Ok(())
}
