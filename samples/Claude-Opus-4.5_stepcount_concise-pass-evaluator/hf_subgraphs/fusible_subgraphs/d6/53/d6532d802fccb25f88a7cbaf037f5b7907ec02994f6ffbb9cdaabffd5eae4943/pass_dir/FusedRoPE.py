import torch
import triton
import triton.language as tl

# Pattern for RoPE: cat -> cos/sin -> scale -> cast to bf16
def pattern(freqs):
    tmp_1 = torch.cat((freqs, freqs), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7

def replacement_args(freqs):
    return (freqs,)

@triton.jit
def fused_rope_kernel(
    freqs_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    inner_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute source index (freqs is half the size in last dim)
    # Output shape: [..., 2*inner_dim], input shape: [..., inner_dim]
    outer_idx = offsets // (2 * inner_dim)
    inner_idx = offsets % (2 * inner_dim)
    src_inner_idx = inner_idx % inner_dim
    src_offset = outer_idx * inner_dim + src_inner_idx
    
    # Load from freqs
    freqs = tl.load(freqs_ptr + src_offset, mask=mask, other=0.0)
    
    # Compute cos and sin (scale by 1.0 is no-op)
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)
    
    # Store as bfloat16
    tl.store(cos_out_ptr + offsets, cos_val.to(tl.bfloat16), mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_rope(freqs):
    # freqs shape: [batch, seq_len, dim] -> output shape: [batch, seq_len, 2*dim]
    shape = freqs.shape
    inner_dim = shape[-1]
    outer_dims = shape[:-1]
    
    # Output shape doubles the last dimension
    out_shape = list(outer_dims) + [2 * inner_dim]
    
    cos_out = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)
    sin_out = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)
    
    n_elements = cos_out.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_rope_kernel[grid](
        freqs,
        cos_out,
        sin_out,
        n_elements,
        inner_dim,
        BLOCK_SIZE,
    )
    
    return cos_out, sin_out

def replacement_func():
    return fused_rope