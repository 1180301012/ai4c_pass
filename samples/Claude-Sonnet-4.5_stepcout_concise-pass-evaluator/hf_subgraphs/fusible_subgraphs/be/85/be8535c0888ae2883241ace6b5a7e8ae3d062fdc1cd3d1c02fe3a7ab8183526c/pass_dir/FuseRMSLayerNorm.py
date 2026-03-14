import torch
import triton
import triton.language as tl


def pattern(in_0, tmp_1):
    """
    Pattern for RMS Layer Normalization (without the residual add).
    Matches: to(float32) + pow(2) + mean + add(eps) + rsqrt + mul + mul
    Takes tmp_1 as input (the result of the residual add).
    """
    tmp_4 = tmp_1.to(torch.float32)
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_1 * tmp_8
    tmp_10 = in_0 * tmp_9
    return tmp_10


def replacement_args(in_0, tmp_1):
    return (in_0, tmp_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def rms_layernorm_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMS LayerNorm kernel.
    Each program handles one row (M dimension).
    """
    row_idx = tl.program_id(0)
    
    # Compute row start offset
    row_start = row_idx * N
    
    # Process the row in blocks
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Accumulator for variance
    var_sum = 0.0
    
    # First pass: compute sum of squares
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = row_start + block_start + cols
        mask = (block_start + cols) < N
        
        # Load input
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Convert to float32 and square
        x_f32 = x.to(tl.float32)
        x_squared = x_f32 * x_f32
        
        # Accumulate
        var_sum += tl.sum(x_squared, axis=0)
    
    # Compute RMS normalization factor
    var = var_sum / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: normalize and apply weight
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = row_start + block_start + cols
        mask = (block_start + cols) < N
        
        # Load input
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize
        x_norm = x * rstd
        
        # Load weight and apply
        weight_offsets = block_start + cols
        weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
        x_out = weight * x_norm
        
        # Store output
        tl.store(out_ptr + offsets, x_out, mask=mask)


@torch.fx.wrap
def fused_rms_layernorm(weight, input_tensor, eps=1e-06):
    """
    Wrapper function for fused RMS LayerNorm.
    
    Args:
        weight: Weight tensor of shape [N]
        input_tensor: Input tensor (tmp_1, the residual sum)
        eps: Epsilon for numerical stability
    
    Returns:
        tmp_10: Normalized and weighted output
    """
    # Get shape info
    M = input_tensor.numel() // input_tensor.shape[-1]  # Number of rows
    N = input_tensor.shape[-1]  # Size of last dimension
    
    # Allocate output
    out = torch.empty_like(input_tensor)
    
    # Launch kernel
    grid = (M,)
    rms_layernorm_kernel[grid](
        input_tensor,
        weight,
        out,
        M,
        N,
        eps,
    )
    
    return out


def replacement_func():
    return fused_rms_layernorm