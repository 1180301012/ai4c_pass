import torch
import triton
import triton.language as tl


# Pattern matching function - matches layer_norm + transpose pattern
def pattern(bias, weight, x):
    """
    Pattern: layer_norm(x, normalized_shape, weight, bias, eps) -> transpose(-1, -2)
    Returns the transposed result.
    """
    ln_out = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)
    transposed = ln_out.transpose(-1, -2)
    return transposed


# Argument extraction function
def replacement_args(bias, weight, x):
    return (bias, weight, x)


# Fully Triton-based LayerNorm + Transpose kernel
@triton.jit
def layer_norm_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, S, H,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm + Transpose kernel in pure Triton.
    Input: [B, S, H] -> Output: [B, H, S]
    """
    # Calculate position
    pid = tl.program_id(0)
    num_elements = B * H * S
    
    if pid >= num_elements:
        return
    
    # Get 3D coordinates for output [B, H, S]
    batch_idx = (pid // (H * S)).to(tl.int32)
    remainder = (pid % (H * S)).to(tl.int32)
    h_idx = (remainder // S).to(tl.int32)
    s_idx = (remainder % S).to(tl.int32)
    
    # Compute base offset for input [B, S, H]
    base_input = (batch_idx * S * H + s_idx * H).to(tl.int64)
    
    # Compute mean: sum all H elements
    sum_val = 0.0
    for hh in range(0, H, BLOCK_SIZE):
        h_offsets = hh + tl.arange(0, BLOCK_SIZE)
        mask = h_offsets < H
        offsets = base_input + h_offsets
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
    
    mean = sum_val / tl.cast(H, tl.float32)
    
    # Compute variance
    sum_sq = 0.0
    for hh in range(0, H, BLOCK_SIZE):
        h_offsets = hh + tl.arange(0, BLOCK_SIZE)
        mask = h_offsets < H
        offsets = base_input + h_offsets
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        diff = vals - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    
    var = sum_sq / tl.cast(H, tl.float32)
    std = tl.sqrt(var + eps)
    
    # Compute output for this element
    for hh in range(0, H, BLOCK_SIZE):
        h_offsets = hh + tl.arange(0, BLOCK_SIZE)
        mask = h_offsets < H
        offsets = base_input + h_offsets
        
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        normalized = (vals - mean) / std
        
        w = tl.load(weight_ptr + h_offsets, mask=mask, other=0.0)
        b_val = tl.load(bias_ptr + h_offsets, mask=mask, other=0.0)
        
        out = normalized * w + b_val
        
        # Store in transposed form [B, H, S]
        out_offsets = (batch_idx * H * S + h_offsets * S + s_idx).to(tl.int64)
        tl.store(output_ptr + out_offsets, out, mask=mask)


@torch.fx.wrap
def layer_norm_transpose_fused(bias, weight, x):
    """Wrapper that launches the fused kernel."""
    B = x.size(0)
    S = x.size(1)
    H = x.size(2)
    eps = 1e-05
    
    out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_elements = B * H * S
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_programs,)
    
    layer_norm_transpose_kernel[grid](
        x, weight, bias, out,
        B, S, H, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return layer_norm_transpose_fused