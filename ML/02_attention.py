"""
Multi-Head Attention - Pure NumPy
Use einsum with suffix notation
"""
import numpy as np
from typing import Tuple, Dict

def forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
            Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
            n_heads: int) -> Tuple[np.ndarray, Dict]:
    """
    Multi-head attention forward
    
    Shapes:
        Q, K, V: (batch, seq_len, d_model)
        Wq, Wk, Wv, Wo: (d_model, d_model)
    
    Hint: einsum notation like 'bsd,dh->bsh'
    """
    # TODO: Implement
    pass

def backward(dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, ...]:
    """Backward pass"""
    # TODO: Implement
    pass

def test():
    np.random.seed(42)
    B, S, D, H = 2, 10, 64, 8
    
    Q = K = V = np.random.randn(B, S, D)
    Wq = Wk = Wv = Wo = np.random.randn(D, D) * 0.1
    
    out, _ = forward(Q, K, V, Wq, Wk, Wv, Wo, H)
    assert out.shape == (B, S, D)
    print("✓ Attention tests passed")

if __name__ == "__main__":
    test()
