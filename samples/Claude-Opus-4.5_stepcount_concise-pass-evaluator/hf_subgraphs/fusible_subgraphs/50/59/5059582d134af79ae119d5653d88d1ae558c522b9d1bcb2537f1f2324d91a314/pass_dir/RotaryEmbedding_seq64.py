import torch
from torch import device
import triton
import triton.language as tl

# Match the cat + to + cos + sin + reshape pattern for more operation fusion
def pattern(outer_result):
    tmp_4 = torch.cat((outer_result, outer_result), dim=-1)
    tmp_5 = tmp_4.to(device(type='cuda', index=0))
    tmp_6 = tmp_5.cos()
    tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_5.sin()
    tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
    return tmp_7, tmp_9

def replacement_args(outer_result):
    return (outer_result,)

# Fused kernel: reads from half-sized input, writes full-sized output
# This avoids the intermediate cat tensor allocation
@triton.jit
def fused_cat_cos_sin_kernel(
    input_ptr,      # [seq_len, half_dim]
    cos_out_ptr,    # [seq_len, full_dim]
    sin_out_ptr,    # [seq_len, full_dim]
    seq_len,
    half_dim,
    full_dim,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Process in 1D
    n_elements = seq_len * full_dim
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Map output position to input position
    row = offsets // full_dim
    col = offsets % full_dim
    input_col = col % half_dim  # Wrap around for cat effect
    
    # Load from input (smaller tensor)
    input_offsets = row * input_stride + input_col
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute cos and sin
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store to output (larger tensor)
    output_offsets = row * output_stride + col
    tl.store(cos_out_ptr + output_offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + output_offsets, sin_val, mask=mask)

@torch.fx.wrap
def _fused_cat_cos_sin(outer_result):
    seq_len, half_dim = outer_result.shape
    full_dim = half_dim * 2
    
    cos_out = torch.empty((seq_len, full_dim), device=outer_result.device, dtype=outer_result.dtype)
    sin_out = torch.empty((seq_len, full_dim), device=outer_result.device, dtype=outer_result.dtype)
    
    n_elements = seq_len * full_dim
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_cat_cos_sin_kernel[grid](
        outer_result,
        cos_out,
        sin_out,
        seq_len,
        half_dim,
        full_dim,
        outer_result.stride(0),
        cos_out.stride(0),
        BLOCK_SIZE,
    )
    
    return cos_out, sin_out

def fused_cat_cos_sin_reshape(outer_result):
    cos_out, sin_out = _fused_cat_cos_sin(outer_result)
    cos_reshaped = cos_out[None, None, slice(None, None, None), slice(None, None, None)]
    sin_reshaped = sin_out[None, None, slice(None, None, None), slice(None, None, None)]
    return cos_reshaped, sin_reshaped

def replacement_func():
    return fused_cat_cos_sin_reshape