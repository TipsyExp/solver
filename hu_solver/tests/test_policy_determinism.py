import os
import shutil
import tempfile
import hashlib

import pytest

try:
    from hu_solver_native import SolverNative
except ImportError:
    pytest.skip("native extension not available", allow_module_level=True)


def sha256_hexdig(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def policy_body_sha256(path: str) -> str:
    """Hash body only (skip 32-byte policy header)."""
    with open(path, "rb") as f:
        head = f.read(32)
        assert len(head) == 32, "policy.bin header too short"
        body = f.read()
    return sha256_hexdig(body)


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_hexdig(f.read())


def train_and_hash(seed: int, iters: int = 5000):
    """Train with a given seed; return (policy_body_sha, index_sha)."""
    tmpdir = tempfile.mkdtemp()
    try:
        s = SolverNative(seed=seed)
        s.train(iters=iters, out_dir=tmpdir)
        policy_bin = os.path.join(tmpdir, "policy.bin")
        index_bin = os.path.join(tmpdir, "index.bin")
        return policy_body_sha256(policy_bin), file_sha256(index_bin)
    finally:
        shutil.rmtree(tmpdir)


def test_policy_determinism_same_seed():
    """Same seed → identical checksums for policy body and index."""
    h1, i1 = train_and_hash(seed=42)
    h2, i2 = train_and_hash(seed=42)
    assert h1 == h2, "policy.bin (body) checksum differs for same seed"
    assert i1 == i2, "index.bin checksum differs for same seed"


def test_policy_determinism_different_seed():
    """Different seeds → at least one of the files must differ."""
    h1, i1 = train_and_hash(seed=1)
