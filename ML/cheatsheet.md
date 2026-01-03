# Quick Reference: Shapes & FLOPs

## MLP

```
, x: (B, D_in)
W1: (D_in, H) → z1 = x @ W1 + b1
a1 = ReLU(z1): (B, H)
W2: (H, D_out) → out = a1 @ W2 + b2

FWD FLOPs: 2*B*D_in*H + 2*B*H*D_out
BWD FLOPs: ~3x FWD
```

## Multi-Head Attention

```
Q, K, V: (B, S, D)
Wq, Wk, Wv: (D, D)

# Project
q = Q @ Wq → (B, S, D)
k = K @ Wk → (B, S, D)
v = V @ Wv → (B, S, D)

# Reshape for heads
q: (B, S, H, D/H) → (B, H, S, D/H)
k, v: same

# Attention
scores = q @ k.T / sqrt(D/H) → (B, H, S, S)
attn = softmax(scores) @ v → (B, H, S, D/H)

# Output projection
out = attn.reshape(B, S, D) @ Wo

FLOPs: 4*B*S*D² + 2*B*H*S²*D/H
     = 4BSD² + 2BS²D
```

## RoPE

```
Apply rotation matrix R(θ) to pairs of dimensions
θ_i = position / 10000^(2i/d)

cos/sin applied to (x[..., 0::2], x[..., 1::2])
```

## Parallelism

```
Tensor Parallel (TP): Split weights across devices
  - Row: split W by rows, allreduce after
  - Column: split W by cols, allgather after

Data Parallel (DP): Same model, split batch
  - AllReduce gradients
```
