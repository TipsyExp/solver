use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};

/// ---- Config (minimal) ----
#[derive(Debug, Deserialize, Clone)]
struct BetSizes {
    #[serde(default)]
    preflop: Option<Vec<serde_yaml::Value>>,
    #[serde(default)]
    flop: Option<Vec<serde_yaml::Value>>,
    #[serde(default)]
    turn: Option<Vec<serde_yaml::Value>>,
    #[serde(default)]
    river: Option<Vec<serde_yaml::Value>>,
}
#[derive(Debug, Deserialize, Clone)]
struct ConfigYaml {
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    bet_sizes: Option<BetSizes>,
}

/// ---- Q4 (×16) quantization with round-half-to-even ----
fn round_half_to_even(x: f64) -> f64 {
    let f = x.floor();
    let r = x - f;
    if r > 0.5 {
        f + 1.0
    } else if r < 0.5 {
        f
    } else {
        if (f as i64) % 2 == 0 { f } else { f + 1.0 }
    }
}
fn q4_quantize(bb: f32) -> u16 {
    let cap = 255.9375_f32;
    let v = bb.clamp(0.0, cap);
    let q = round_half_to_even((v as f64) * 16.0) as i64;
    q.max(0).min(4095) as u16 // 12 bits
}
fn q4_dequantize(q: u16) -> f32 {
    (q as f32) / 16.0
}

/// ---- BetStateId v1 (Q4) ----
/// to_call_q:12 | last_raise_q:12 | jam:1 | pos:1 | street:2 | reserved:4
fn pack_bet_state_id(street: u8, pos: u8, to_call_bb: f32, last_raise_bb: f32, jam: bool) -> u32 {
    let toq = q4_quantize(to_call_bb) as u32;
    let lrq = q4_quantize(last_raise_bb) as u32;
    let jam_flag = if jam { 1u32 } else { 0u32 };
    let street2 = (street as u32) & 0b11;
    let pos1 = (pos as u32) & 0b1;
    (toq & 0xFFF)
        | ((lrq & 0xFFF) << 12)
        | (jam_flag << 24)
        | (pos1 << 25)
        | (street2 << 26)
    // top 4 bits reserved = 0
}

/// ---- NodeId 64-bit ----
/// board_bucket:11 | hand_bucket:11 | bet_state_id:32 | street:2 | reserved:8
fn pack_node_id(board_bucket: u16, hand_bucket: u16, bet_state_id: u32, street: u8) -> u64 {
    let bb = (board_bucket as u64) & 0x7FF;
    let hb = (hand_bucket as u64) & 0x7FF;
    let bsi = (bet_state_id as u64) & 0xFFFF_FFFF;
    let st = (street as u64) & 0x3;
    (bb << 53) | (hb << 42) | (bsi << 10) | (st << 8)
}

/// simple splitmix64 for deterministic per-iteration seeds
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// global fixed actions (street-specific mapping documented in meta):
/// indices: 0=fold, 1=check/call, 2=size A, 3=size B, 4=size C, 5=jam
const ACTIONS_PER_NODE: usize = 6;

/// small HU subgame shape for smoke:
const FLOP_BUCKETS: u16 = 4;
const HAND_BUCKETS: u16 = 8;

#[pyclass]
struct SolverNative {
    seed: u64,
    // discrete sizes by street (defaults if not in config)
    preflop_sizes: Vec<String>, // ["2.5x", "8x", "jam"]
    flop_sizes: Vec<String>,    // ["0.33p","0.75p","1.25p","jam"]
    // mapping: NodeId -> row index (sorted)
    nodes: BTreeMap<u64, usize>,
    // regrets/avg
    regrets: Vec<f32>,      // rows * ACTIONS_PER_NODE
    strategy_sum: Vec<f32>, // rows * ACTIONS_PER_NODE
    // policy snapshot saved on save_policy()
    policy_rows: Vec<f32>, // flattened per-row distribution
}

#[pymethods]
impl SolverNative {
    #[new]
    fn new(seed: Option<u64>) -> Self {
        let mut s = Self {
            seed: seed.unwrap_or(1234),
            preflop_sizes: vec!["2.5x".to_string(), "8x".to_string(), "jam".to_string()],
            flop_sizes: vec!["0.33p".to_string(), "0.75p".to_string(), "1.25p".to_string(), "jam".to_string()],
            nodes: BTreeMap::new(),
            regrets: Vec::new(),
            strategy_sum: Vec::new(),
            policy_rows: Vec::new(),
        };
        s.init_nodes();
        s
    }

    /// Optional: load config YAML if you want sizes from file (seed too).
    fn load_config_yaml(&mut self, config_path: &str) -> PyResult<()> {
        let mut f = File::open(config_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open config: {e}")))?;
        let mut buf = String::new();
        f.read_to_string(&mut buf).unwrap();
        let cfg: ConfigYaml = serde_yaml::from_str(&buf)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("parse yaml: {e}")))?;
        if let Some(sd) = cfg.seed {
            self.seed = sd;
        }
        if let Some(bs) = cfg.bet_sizes {
            if let Some(pf) = bs.preflop {
                self.preflop_sizes = pf.iter().map(|v| yaml_val_to_size(v)).collect();
            }
            if let Some(ff) = bs.flop {
                self.flop_sizes = ff.iter().map(|v| yaml_val_to_size(v)).collect();
            }
        }
        Ok(())
    }

    /// minimal ES-MCCFR + CFR+ over a tiny HU subgame (preflop+flop)
    /// default iters kept CI-small.
    #[pyo3(signature = (iters=None, out_dir))]
    fn train(&mut self, iters: Option<u64>, out_dir: &str) -> PyResult<()> {
        let iters = iters.unwrap_or(120_000);
        create_dir_all(out_dir).ok();
        for i in 0..iters {
            let iter_seed = splitmix64(self.seed ^ (i + 1));
            let mut rng = StdRng::seed_from_u64(iter_seed);
            let (row, legal_mask) = self.sample_row_and_mask(&mut rng);

            // regret-matching policy from current regrets (clamped >=0)
            let base = row * ACTIONS_PER_NODE;
            let mut pos_reg = [0f32; ACTIONS_PER_NODE];
            for a in 0..ACTIONS_PER_NODE {
                let r = self.regrets[base + a].max(0.0);
                pos_reg[a] = if legal_mask[a] { r } else { 0.0 };
            }
            let sum_r: f32 = pos_reg.iter().sum();
            let mut sigma = [0f32; ACTIONS_PER_NODE];
            if sum_r > 0.0 {
                for a in 0..ACTIONS_PER_NODE {
                    sigma[a] = pos_reg[a] / sum_r;
                }
            } else {
                // uniform over legal
                let k = legal_mask.iter().filter(|&&b| b).count().max(1) as f32;
                for a in 0..ACTIONS_PER_NODE {
                    sigma[a] = if legal_mask[a] { 1.0 / k } else { 0.0 };
                }
            }

            // deterministic toy payoff for convergence (non-uniform, stable)
            let mut v = [0f32; ACTIONS_PER_NODE];
            for a in 0..ACTIONS_PER_NODE {
                if legal_mask[a] {
                    let base_val = ((row as f32 % 13.0) * 0.01) + (a as f32) * 0.001;
                    v[a] = base_val;
                } else {
                    v[a] = 0.0;
                }
            }
            let u: f32 = (0..ACTIONS_PER_NODE).map(|a| sigma[a] * v[a]).sum();

            // CFR+ cumulative regrets
            for a in 0..ACTIONS_PER_NODE {
                if legal_mask[a] {
                    let idx = base + a;
                    let r = self.regrets[idx] + (v[a] - u);
                    self.regrets[idx] = r.max(0.0);
                }
            }
            // accumulate average strategy (linear averaging)
            for a in 0..ACTIONS_PER_NODE {
                self.strategy_sum[base + a] += sigma[a];
            }

            if i % 50_000 == 0 {
                let mut mf = File::options()
                    .create(true)
                    .append(true)
                    .open(format!("{out_dir}/metrics.csv"))
                    .unwrap();
                let l2: f32 = self.regrets.iter().map(|x| x * x).sum::<f32>().sqrt();
                writeln!(mf, "{},{}", i, l2).ok();
            }
        }
        Ok(())
    }

    /// Save policy.bin (v1, 32-byte header), index.bin (NodeId->offset),
    /// and meta.json with checksums and discrete_sizes.
    fn save_policy(&mut self, out_dir: &str) -> PyResult<()> {
        create_dir_all(out_dir).ok();

        // 1) Build average policy rows from strategy_sum
        let rows = self.nodes.len();
        self.policy_rows.resize(rows * ACTIONS_PER_NODE, 0.0);
        for (_nid, row) in self.nodes.iter() {
            let base = *row * ACTIONS_PER_NODE;
            let slice = &self.strategy_sum[base..base + ACTIONS_PER_NODE];
            let s: f32 = slice.iter().sum();
            let out = &mut self.policy_rows[base..base + ACTIONS_PER_NODE];
            if s > 0.0 {
                for a in 0..ACTIONS_PER_NODE {
                    out[a] = slice[a] / s;
                }
            } else {
                for a in 0..ACTIONS_PER_NODE {
                    out[a] = 1.0 / (ACTIONS_PER_NODE as f32);
                }
            }
        }

        // 2) policy.bin header (32 bytes)
        let mut pol = File::create(format!("{out_dir}/policy.bin"))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("policy.bin: {e}")))?;
        // magic:u32 'PPOL'=0x50504F4C, version:u16=1, endian:u8=1, actions_mode:u8=0,
        // dtype:u8=1(f32), reserved0:u8=0, actions_per_node:u16, header_size:u32=32,
        // nodes:u64, body_checksum:u32=0, reserved1:u32=0
        let magic: u32 = 0x50504F4C;
        let version: u16 = 1;
        let endian: u8 = 1;
        let actions_mode: u8 = 0;
        let dtype: u8 = 1;
        let reserved0: u8 = 0;
        let actions_per_node: u16 = ACTIONS_PER_NODE as u16;
        let header_size: u32 = 32;
        let nodes_u64: u64 = rows as u64;
        let body_checksum: u32 = 0;
        let reserved1: u32 = 0;
        pol.write_all(&magic.to_le_bytes())?;
        pol.write_all(&version.to_le_bytes())?;
        pol.write_all(&[endian, actions_mode, dtype, reserved0])?;
        pol.write_all(&actions_per_node.to_le_bytes())?;
        pol.write_all(&header_size.to_le_bytes())?;
        pol.write_all(&nodes_u64.to_le_bytes())?;
        pol.write_all(&body_checksum.to_le_bytes())?;
        pol.write_all(&reserved1.to_le_bytes())?;

        // 3) body: flattened float32 rows
        let mut body_bytes = Vec::<u8>::with_capacity(self.policy_rows.len() * 4);
        for x in &self.policy_rows {
            body_bytes.extend_from_slice(&x.to_le_bytes());
        }
        pol.write_all(&body_bytes)?;

        // 4) index.bin: (node_id: u64, offset: u64) sorted by node_id
        let mut idxf = File::create(format!("{out_dir}/index.bin"))?;
        let mut offset: u64 = 0; // bytes into policy body
        for (node_id, _row) in &self.nodes {
            idxf.write_all(&node_id.to_le_bytes())?;
            idxf.write_all(&offset.to_le_bytes())?;
            offset += (ACTIONS_PER_NODE * 4) as u64; // 4 bytes per float
        }

        // 5) checksums
        let mut pol_f = File::open(format!("{out_dir}/policy.bin"))?;
        let mut pol_buf = Vec::new();
        pol_f.read_to_end(&mut pol_buf).ok();
        let pol_sha = Sha256::digest(&pol_buf);
        let pol_hex = hex::encode(pol_sha);

        let mut idx_f = File::open(format!("{out_dir}/index.bin"))?;
        let mut idx_buf = Vec::new();
        idx_f.read_to_end(&mut idx_buf).ok();
        let idx_sha = Sha256::digest(&idx_buf);
        let idx_hex = hex::encode(idx_sha);

        // 6) meta.json
        let meta = json!({
          "schema_version": 1,
          "training": {
            "seed": self.seed,
            "iters": 0,
            "threads": 1
          },
          "bucket_counts": {"board": {"flop": FLOP_BUCKETS}, "hand": HAND_BUCKETS},
          "bet_state_id": {"version": 1, "scale_bb_q": 16, "caps": {"to_call_bb": 255.9375, "last_raise_bb": 255.9375}},
          "node_id_bits": {"board_bucket": 11, "hand_bucket": 11, "bet_state_id": 32, "street": 2},
          "discrete_sizes": {
            "preflop": self.preflop_sizes,
            "flop": self.flop_sizes
          },
          "policy_header": {"version": 1, "dtype": "f32", "actions_mode": "fixed", "actions_per_node": ACTIONS_PER_NODE},
          "index": {"format": "node_id", "entry": "fixed", "sorted": true},
          "checksum": {"policy_bin_sha256": pol_hex, "index_bin_sha256": idx_hex},
          "total_nodes": self.nodes.len()
        });
        let mut mf = File::create(format!("{out_dir}/meta.json"))?;
        mf.write_all(serde_json::to_string_pretty(&meta).unwrap().as_bytes())?;

        Ok(())
    }

    fn load_policy(&mut self, out_dir: &str) -> PyResult<()> {
        let mut f = File::open(format!("{out_dir}/policy.bin"))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        let mut head = [0u8; 32];
        f.read_exact(&mut head)?;
        // minimal header parse
        let _magic = u32::from_le_bytes(head[0..4].try_into().unwrap());
        let _version = u16::from_le_bytes(head[4..6].try_into().unwrap());
        let _endian = head[6];
        let _actions_mode = head[7];
        let _dtype = head[8];
        let _res0 = head[9];
        let actions_per_node = u16::from_le_bytes(head[10..12].try_into().unwrap());
        let _header_size = u32::from_le_bytes(head[12..16].try_into().unwrap());
        let nodes = u64::from_le_bytes(head[16..24].try_into().unwrap()) as usize;

        if actions_per_node as usize != ACTIONS_PER_NODE {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "actions_per_node mismatch",
            ));
        }
        let body_len = nodes * ACTIONS_PER_NODE * 4;
        let mut body = vec![0u8; body_len];
        f.read_exact(&mut body)?;

        self.policy_rows.resize(nodes * ACTIONS_PER_NODE, 0.0);
        let mut j = 0;
        for i in 0..(nodes * ACTIONS_PER_NODE) {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(&body[j..j + 4]);
            j += 4;
            self.policy_rows[i] = f32::from_le_bytes(arr);
        }

        if self.nodes.is_empty() {
            self.init_nodes();
        }
        Ok(())
    }

    /// query(state_dict) -> {action: prob} using saved/avg policy rows
    fn query(&self, py: Python<'_>, state: &PyDict) -> PyResult<PyObject> {
        // Force PyDict::get_item (returns Option<&PyAny>) to avoid PyAny::get_item (Result).
        let street: u8 = PyDict::get_item(state, "street")
            .ok_or_else(|| err("missing street"))?
            .extract()?;
        let to_call: f32 = PyDict::get_item(state, "to_call")
            .ok_or_else(|| err("missing to_call"))?
            .extract()?;
        let last_raise: f32 = PyDict::get_item(state, "last_raise")
            .ok_or_else(|| err("missing last_raise"))?
            .extract()?;
        let pos: u8 = PyDict::get_item(state, "pos")
            .ok_or_else(|| err("missing pos"))?
            .extract()?;

        let jam_legal = true; // documented rule: jam allowed when stack permits (assumed true in tiny slice)
        let bsi = pack_bet_state_id(street, pos, to_call, last_raise, jam_legal);

        // optional buckets (default 0)
        let board_bucket: u16 = PyDict::get_item(state, "board_bucket")
            .and_then(|o| o.extract::<u16>().ok())
            .unwrap_or(0);
        let hand_bucket: u16 = PyDict::get_item(state, "hand_bucket")
            .and_then(|o| o.extract::<u16>().ok())
            .unwrap_or(0);

        let nid = pack_node_id(board_bucket, hand_bucket, bsi, street);

        // resolve row (fallback to 0 if not present)
        let row = self.nodes.get(&nid).copied().unwrap_or(0);
        let base = row * ACTIONS_PER_NODE;

        // legal mask by street
        let mut legal = [false; ACTIONS_PER_NODE];
        legal[0] = true; // fold
        legal[1] = true; // check/call
        match street {
            0 => { // preflop
                legal[2] = true; // size A (2.5x)
                legal[3] = true; // size B (8x)
                legal[4] = false; // size C unused preflop
                legal[5] = true; // jam
            }
            1 => { // flop
                legal[2] = true; // 0.33p
                legal[3] = true; // 0.75p
                legal[4] = true; // 1.25p
                legal[5] = true; // jam
            }
            _ => {}
        }

        // read row; mask and renormalize
        let mut probs = [0f32; ACTIONS_PER_NODE];
        let mut sum = 0f32;
        for a in 0..ACTIONS_PER_NODE {
            let p = self.policy_rows.get(base + a).copied().unwrap_or(0.0);
            let v = if legal[a] { p } else { 0.0 };
            probs[a] = v;
            sum += v;
        }
        if sum <= 0.0 {
            let k = legal.iter().filter(|&&b| b).count().max(1) as f32;
            for a in 0..ACTIONS_PER_NODE {
                probs[a] = if legal[a] { 1.0 / k } else { 0.0 };
            }
        } else {
            for a in 0..ACTIONS_PER_NODE {
                probs[a] /= sum;
            }
        }

        let out = PyDict::new(py);
        out.set_item("fold", probs[0])?;
        out.set_item("check_call", probs[1])?;
        out.set_item("size_A", probs[2])?;
        out.set_item("size_B", probs[3])?;
        out.set_item("size_C", probs[4])?;
        out.set_item("jam", probs[5])?;
        Ok(out.into())
    }
}

/// ---- helpers ----
fn yaml_val_to_size(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => n.to_string(),
        _ => "jam".to_string(),
    }
}
fn err(msg: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.to_string())
}

impl SolverNative {
    fn init_nodes(&mut self) {
        // Build tiny HU slice: preflop (street=0) and flop (street=1),
        // FLOP_BUCKETS × HAND_BUCKETS × 2 positions × small BetStateId grid.
        // Grid: to_call_q in {0, 8}, last_raise_q in {0, 16}, jam in {0,1}, pos in {0,1}.
        let mut pairs: Vec<(u64, usize)> = Vec::new();
        let mut row = 0usize;

        for &street in &[0u8, 1u8] {
            for pos in 0u8..=1 {
                for bb in 0..FLOP_BUCKETS {
                    for hb in 0..HAND_BUCKETS {
                        for toq in [0u16, 8u16] {
                            for lrq in [0u16, 16u16] {
                                for jam in [false, true] {
                                    let bsi = ((toq as u32) & 0xFFF)
                                        | (((lrq as u32) & 0xFFF) << 12)
                                        | ((jam as u32) << 24)
                                        | (((pos as u32) & 0x1) << 25)
                                        | (((street as u32) & 0x3) << 26);
                                    let nid = pack_node_id(bb, hb, bsi, street);
                                    pairs.push((nid, row));
                                    row += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        self.nodes = pairs.into_iter().collect();

        let rows = self.nodes.len();
        self.regrets = vec![0.0; rows * ACTIONS_PER_NODE];
        self.strategy_sum = vec![0.0; rows * ACTIONS_PER_NODE];
        self.policy_rows = vec![0.0; rows * ACTIONS_PER_NODE];
    }

    fn sample_row_and_mask(&self, rng: &mut StdRng) -> (usize, [bool; ACTIONS_PER_NODE]) {
        let rows = self.nodes.len();
        let idx = (rng.gen::<u64>() as usize) % rows;
        let node_id = self.nodes.iter().nth(idx).unwrap().0;
        let street = ((node_id >> 8) & 0x3) as u8;
        let mut mask = [false; ACTIONS_PER_NODE];
        mask[0] = true;
        mask[1] = true;
        if street == 0 {
            mask[2] = true;
            mask[3] = true;
            mask[4] = false;
            mask[5] = true;
        } else {
            mask[2] = true;
            mask[3] = true;
            mask[4] = true;
            mask[5] = true;
        }
        (idx, mask)
    }
}

/// ---- module export ----
#[pymodule]
fn hu_solver_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SolverNative>()?;
    Ok(())
}
