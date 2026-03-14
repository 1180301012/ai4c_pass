import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Pattern matching cat + batch_norm + prelu"""
    tmp_5 = torch.cat([in_5, in_6], 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_7 = torch.prelu(tmp_6, in_0)
    return tmp_7

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_cat_bn_prelu_kernel(
    in_5_ptr, in_6_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    prelu_weight_ptr,
    out_ptr,
    N, C, H, W,
    C_half: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for cat + batch_norm + prelu"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N * C * H * W
    
    # Calculate indices
    n_idx = offsets // (C * H * W)
    c_idx = (offsets // (H * W)) % C
    hw_idx = offsets % (H * W)
    
    # Determine if we're in first or second half of channels (for cat)
    is_first_half = c_idx < C_half
    c_orig = tl.where(is_first_half, c_idx, c_idx - C_half)
    
    # Calculate source offsets
    src_offset = n_idx * C_half * H * W + c_orig * H * W + hw_idx
    
    # Load from appropriate input tensor
    x = tl.where(is_first_half,
                 tl.load(in_5_ptr + src_offset, mask=mask, other=0.0),
                 tl.load(in_6_ptr + src_offset, mask=mask, other=0.0))
    
    # Load batch norm parameters
    bn_mean = tl.load(running_mean_ptr + c_idx, mask=mask, other=0.0)
    bn_var = tl.load(running_var_ptr + c_idx, mask=mask, other=0.0)
    bn_weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    bn_bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # Apply batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - bn_mean) / tl.sqrt(bn_var + eps)
    bn_out = normalized * bn_weight + bn_bias
    
    # Load prelu weight and apply prelu
    prelu_w = tl.load(prelu_weight_ptr + c_idx, mask=mask, other=0.0)
    out = tl.where(bn_out > 0, bn_out, bn_out * prelu_w)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_cat_bn_prelu(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """Wrapper for fused cat + batch_norm + prelu kernel"""
    N, C_half, H, W = in_5.shape
    C = C_half * 2
    
    out = torch.empty((N, C, H, W), device=in_5.device, dtype=in_5.dtype)
    
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_cat_bn_prelu_kernel[grid](
        in_5, in_6,
        in_1, in_2,
        in_4, in_3,
        in_0,
        out,
        N, C, H, W,
        C_half=C_half,
        eps=0.001,
    )
    
    return out

def replacement_func():
    return fused_cat_bn_prelu