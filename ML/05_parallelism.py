"""
Parallelism Strategies - Pure NumPy
Row-sharded, Column-parallel, Data-parallel MLP
"""
import numpy as np
from typing import List, Tuple

# =============================================================================
# Row-Sharded (Tensor Parallel) MLP
# =============================================================================

def row_shard_forward(x: np.ndarray, W_shards: List[np.ndarray], 
                      b: np.ndarray) -> np.ndarray:
    """
    Row-sharded linear: W split by rows across devices
    Each shard: (in_features, hidden/n_devices)
    Requires AllReduce after
    """
    # TODO: Implement
    pass

# =============================================================================
# Column-Parallel MLP  
# =============================================================================

def col_parallel_forward(x: np.ndarray, W_shards: List[np.ndarray],
                         b_shards: List[np.ndarray]) -> List[np.ndarray]:
    """
    Column-parallel linear: W split by columns
    Each shard: (in_features/n_devices, hidden)
    Requires AllGather after
    """
    # TODO: Implement
    pass

# =============================================================================
# Data-Parallel MLP
# =============================================================================

def data_parallel_forward(x_shards: List[np.ndarray], W: np.ndarray, 
                          b: np.ndarray) -> List[np.ndarray]:
    """
    Data-parallel: same model, split batch
    Each x_shard: (batch/n_devices, features)
    """
    # TODO: Implement
    pass

def data_parallel_backward(dout_shards: List[np.ndarray], 
                           caches: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data-parallel backward: compute local grads, AllReduce
    Returns: dW, db (averaged across devices)
    """
    # TODO: Implement
    pass

def test():
    np.random.seed(42)
    B, D_in, D_out = 8, 16, 32
    n_devices = 4
    
    x = np.random.randn(B, D_in)
    W = np.random.randn(D_in, D_out) * 0.1
    b = np.zeros(D_out)
    
    # Test data parallel
    x_shards = np.array_split(x, n_devices)
    outs = data_parallel_forward(x_shards, W, b)
    out = np.concatenate(outs, axis=0)
    assert out.shape == (B, D_out)
    
    print("✓ Parallelism tests passed")

if __name__ == "__main__":
    test()
