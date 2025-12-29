"""
KV Cache for Inference - Pure NumPy
"""
import numpy as np
from typing import Tuple

class KVCache:
    """Simple KV cache for autoregressive inference"""
    
    def __init__(self, max_seq: int, n_heads: int, head_dim: int):
        """Initialize empty cache"""
        # TODO: Implement
        pass
    
    def update(self, k: np.ndarray, v: np.ndarray, start_pos: int) -> None:
        """
        Update cache with new k, v starting at position
        
        Shapes:
            k, v: (batch, new_seq_len, n_heads, head_dim)
        """
        # TODO: Implement
        pass
    
    def get(self, end_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached k, v from 0 to end_pos"""
        # TODO: Implement
        pass

def test():
    B, S, H, D = 1, 32, 8, 64
    cache = KVCache(S, H, D)
    
    # Prefill
    k1 = np.random.randn(B, 10, H, D)
    v1 = np.random.randn(B, 10, H, D)
    cache.update(k1, v1, 0)
    
    # Decode step
    k2 = np.random.randn(B, 1, H, D)
    v2 = np.random.randn(B, 1, H, D)
    cache.update(k2, v2, 10)
    
    k_full, v_full = cache.get(11)
    assert k_full.shape == (B, 11, H, D)
    print("✓ KV Cache tests passed")

if __name__ == "__main__":
    test()
