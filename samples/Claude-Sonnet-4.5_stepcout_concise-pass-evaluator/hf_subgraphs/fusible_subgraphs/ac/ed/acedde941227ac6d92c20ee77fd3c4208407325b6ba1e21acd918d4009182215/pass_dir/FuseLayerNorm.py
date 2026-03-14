import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    """Pattern to match layer normalization"""
    tmp_11 = torch.nn.functional.layer_norm(in_3, (2560,), in_1, in_0, 1e-05)
    return tmp_11

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized layer normalization kernel"""
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    
    # Compute mean
    mean = 0.0
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    
    # Compute variance
    var = 0.0
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    
    # Compute normalized output
    rstd = 1.0 / tl.sqrt(var + eps)
    
    for i in range(0, N, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        x_norm = (x - mean) * rstd
        # Apply affine transformation
        out = x_norm * weight + bias
        
        tl.store(out_ptr + row_start + offsets, out.to(tl.float16), mask=mask)

@torch.fx.wrap
def fused_layer_norm(x, weight, bias):
    """Optimized layer normalization"""
    # x shape: [batch, seq_len, hidden_dim]
    # weight, bias shape: [hidden_dim]
    
    M = x.numel() // x.shape[-1]  # Number of rows
    N = x.shape[-1]  # Hidden dimension
    
    out = torch.empty_like(x)
    eps = 1e-05
    
    # Launch kernel - one program per row
    grid = (M,)
    
    layer_norm_kernel[grid](
        x,
        weight,
        bias,
        out,
        N,
        eps,
    )
    
    return out

def replacement_func():
    return fused_layer_norm