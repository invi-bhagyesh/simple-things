"""
RoPE (Rotary Position Embeddings) - Pure NumPy
"""
import numpy as np

def get_freqs(dim: int, max_seq: int, base: float = 10000.0) -> np.ndarray:
    """
    Compute rotation frequencies
    θ_i = 1 / base^(2i/dim)
    """
    # TODO: Implement
    pass

def apply_rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Apply rotary embeddings to x
    
    Shapes:
        x: (batch, seq_len, n_heads, head_dim)
        positions: (seq_len,)
    """
    # TODO: Implement
    # Hint: apply cos/sin rotation to pairs (x[..., 0::2], x[..., 1::2])
    pass

def test():
    np.random.seed(42)
    B, S, H, D = 2, 16, 8, 64
    
    x = np.random.randn(B, S, H, D)
    pos = np.arange(S)
    
    out = apply_rope(x, pos)
    assert out.shape == x.shape
    print("✓ RoPE tests passed")

if __name__ == "__main__":
    test()
