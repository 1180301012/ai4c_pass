import torch
import triton
import triton.language as tl

# Pattern to match: layer_norm with normalized_shape=(2560,)
def pattern(x, weight, bias):
    out = torch.nn.functional.layer_norm(x, (2560,), weight, bias, 1e-05)
    return out

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,  # normalized dimension (2560)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one [batch, seq] position)
    row_idx = tl.program_id(0)
    
    # Pointer to start of this row
    row_start = row_idx * N
    
    # Compute mean and variance in two passes
    # First pass: compute mean
    mean = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N
    
    # Second pass: compute variance
    var = tl.zeros([1], dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    
    # Compute normalization factor
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Third pass: normalize and apply affine transform
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        x_norm = (x - mean) * rstd
        
        # Apply affine transform
        out = x_norm * w + b
        
        # Store result
        tl.store(out_ptr + row_start + offsets, out.to(tl.float16), mask=mask)


@torch.fx.wrap
def layer_norm_fused(x, weight, bias):
    # x shape: [batch, seq_len, hidden_dim]
    # weight, bias shape: [hidden_dim]
    original_shape = x.shape
    N = original_shape[-1]  # 2560
    
    # Flatten to 2D for easier processing
    x_2d = x.reshape(-1, N)
    num_rows = x_2d.shape[0]
    
    # Output tensor
    out = torch.empty_like(x_2d)
    
    # Launch kernel - one program per row
    layer_norm_kernel[(num_rows,)](
        x_2d,
        weight,
        bias,
        out,
        N,
        1e-05,
    )
    
    return out.reshape(original_shape)


def replacement_func():
    return layer_norm_fused