import torch
import triton
import triton.language as tl

def pattern(tmp_13, in_3, in_2):
    return torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)

def replacement_args(tmp_13, in_3, in_2):
    return (tmp_13, in_3, in_2)

@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr,
    Y_ptr,
    stride_x, stride_w, stride_b,
    stride_y,
    N, D,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    row_ids = row_start + tl.arange(0, BLOCK_N)
    row_mask = row_ids < N

    col_ids = tl.arange(0, BLOCK_D)
    col_mask = col_ids < D

    # Load X data
    x = tl.load(
        X_ptr + row_ids[:, None] * stride_x + col_ids[None, :] * 1,
        mask=(row_mask[:, None] & col_mask[None, :]),
        other=0.0
    )

    # Compute mean over channel dimension
    mean = tl.sum(x, axis=1) / D
    # Compute variance over channel dimension
    var = tl.sum((x - mean[:, None]) ** 2, axis=1) / D
    inv_var = 1.0 / tl.sqrt(var + eps)

    # Load weight and bias
    w = tl.load(W_ptr + col_ids, mask=col_mask, other=0.0)
    b = tl.load(B_ptr + col_ids, mask=col_mask, other=0.0)

    # Apply layer norm
    y = (x - mean[:, None]) * inv_var[:, None] * w + b

    # Store result
    tl.store(
        Y_ptr + row_ids[:, None] * stride_y + col_ids[None, :] * 1,
        y,
        mask=(row_mask[:, None] & col_mask[None, :])
    )

@torch.fx.wrap
def layer_norm_wrapper(X, W, B):
    batch, seq_len, channels = X.shape
    N = batch * seq_len
    
    # Reshape to [N, channels] for kernel processing
    X_flat = X.reshape(-1, channels)
    Y_flat = torch.empty_like(X_flat)
    
    # Kernel configuration
    BLOCK_N = 64
    BLOCK_D = 256
    grid = (triton.cdiv(N, BLOCK_N),)

    # Launch kernel
    layer_norm_kernel[grid](
        X_flat,
        W,
        B,
        Y_flat,
        X_flat.stride(0),  # Stride between rows
        W.stride(0),       # Stride for W (1)
        B.stride(0),       # Stride for B (1)
        Y_flat.stride(0),  # Stride between rows
        N,
        channels,
        1e-05,
        BLOCK_N,
        BLOCK_D
    )
    
    # Reshape back to original shape
    return Y_flat.reshape(batch, seq_len, channels)

def replacement_func():
    return layer_norm_wrapper