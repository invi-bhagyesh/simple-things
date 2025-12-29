"""
2-Layer MLP with ReLU - Pure NumPy
FWD + BWD implementation
"""
import numpy as np
from typing import Tuple, Dict

def forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
            W2: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Forward: x -> Linear -> ReLU -> Linear -> out
    
    Shapes:
        x: (batch, in_features)
        W1: (in_features, hidden), b1: (hidden,)
        W2: (hidden, out_features), b2: (out_features,)
    """
    # TODO: Implement
    pass

def backward(dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, ...]:
    """
    Backward pass
    Returns: dx, dW1, db1, dW2, db2
    """
    # TODO: Implement
    pass

def test():
    np.random.seed(42)
    B, D_in, H, D_out = 4, 8, 16, 3
    
    x = np.random.randn(B, D_in)
    W1, b1 = np.random.randn(D_in, H) * 0.1, np.zeros(H)
    W2, b2 = np.random.randn(H, D_out) * 0.1, np.zeros(D_out)
    
    out, cache = forward(x, W1, b1, W2, b2)
    assert out.shape == (B, D_out)
    
    dout = np.random.randn(B, D_out)
    dx, dW1, db1, dW2, db2 = backward(dout, cache)
    assert dx.shape == x.shape
    print("✓ MLP tests passed")

if __name__ == "__main__":
    test()
