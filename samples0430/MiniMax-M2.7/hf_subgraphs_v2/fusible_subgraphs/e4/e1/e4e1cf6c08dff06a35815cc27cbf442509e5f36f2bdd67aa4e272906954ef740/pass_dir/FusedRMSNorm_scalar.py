import torch
import triton
import triton.language as tl

# Pattern matching function for RMSNorm (starting from tmp_2)
def pattern(tmp_2, in_1):
    """
    Pattern matching the RMSNorm computation starting from tmp_2.
    
    The computation pattern is:
    1. tmp_4 = tmp_2.float() - convert to float32
    2. tmp_5 = tmp_4.pow(2) - square
    3. tmp_6 = tmp_5.mean(-1, keepdim=True) - reduce last dim
    4. tmp_7 = tmp_6 + 1e-06 - add epsilon
    5. tmp_8 = torch.rsqrt(tmp_7) - reciprocal sqrt
    6. tmp_9 = tmp_4 * tmp_8 - normalize
    7. tmp_10 = in_1.float() - weight to float32
    8. tmp_11 = 1.0 + tmp_10 - add 1 to weight (bias)
    9. tmp_12 = tmp_9 * tmp_11 - apply weight
    10. tmp_13 = tmp_12.type_as(tmp_2) - cast to original dtype
    """
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim = True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_13

# Argument extraction function
def replacement_args(tmp_2, in_1):
    return (tmp_2, in_1)

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x: tl.int32,
    stride_w: tl.int32,
    stride_out: tl.int32,
    n_elements: tl.int32,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm kernel with autotuning support.
    
    Computes: out = x / sqrt(mean(x^2) + eps) * (1 + weight)
    
    Grid: Each program handles one position in [B, S] dimensions.
    Each program processes BLOCK_SIZE elements along the hidden dimension.
    """
    pid = tl.program_id(0)
    
    # Compute base offset for this position
    base_offset = pid * stride_x
    
    # Compute offsets for this program
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input as bfloat16 and convert to float32
    x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute squared values
    sq = x * x
    
    # Reduce along last dimension (axis=0 for the block dimension)
    sq_sum = tl.sum(sq, axis=0)
    
    # Compute normalized value
    sq_mean = sq_sum / n_elements
    denom = tl.sqrt(sq_mean + EPS)
    norm = x / denom
    
    # Load weight (bfloat16) and convert to float32
    w = tl.load(w_ptr + offsets * stride_w, mask=mask, other=0.0).to(tl.float32)
    out = norm * (w + 1.0)
    
    # Store output (cast to bfloat16)
    out_base_offset = pid * stride_out
    out_bf16 = out.to(tl.bfloat16)
    tl.store(out_ptr + out_base_offset + offsets, out_bf16, mask=mask)

# Autotune configurations
autotune_configs = [
    triton.Config({'BLOCK_SIZE': 2048}, num_stages=1, num_warps=8),
]

@torch.fx.wrap
def rmsnorm_kernel_wrapper(tmp_2, in_1):
    """
    Wrapper for the fused RMSNorm kernel with autotuning.
    
    Args:
        tmp_2: Input tensor after scalar multiplication [B, S, H]
        in_1: RMSNorm weight tensor [H]
    
    Returns:
        normalized_output
    """
    B, S, H = tmp_2.shape
    n_positions = B * S  # Grid size
    EPS = 1e-06
    
    # Create output tensor
    orig_dtype = tmp_2.dtype
    device = tmp_2.device
    
    # tmp_13: normalized output [B, S, H]
    out = torch.empty((B, S, H), dtype=orig_dtype, device=device)
    
    # Compute strides for 3D tensor (row-major)
    stride_x = S * H
    stride_out = S * H
    stride_w = 1  # 1D tensor
    
    # Grid configuration
    grid = (n_positions,)
    
    # Launch kernel with autotuning
    rmsnorm_kernel_autotuned[grid](
        x_ptr=tmp_2,
        w_ptr=in_1,
        out_ptr=out,
        stride_x=stride_x,
        stride_w=stride_w,
        stride_out=stride_out,
        n_elements=H,
        EPS=EPS,
    )
    
    return out

# Autotuned kernel
rmsnorm_kernel_autotuned = triton.autotune(
    configs=autotune_configs,
    key=['n_elements'],
)(rmsnorm_kernel)

def replacement_func():
    return rmsnorm_kernel_wrapper