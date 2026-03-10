import torch
import triton
import triton.language as tl

def pattern(tmp_4, in_5, in_6):
    tmp_5 = tmp_4 * in_6
    tmp_6 = in_5 + tmp_5
    return tmp_6

def replacement_args(tmp_4, in_5, in_6):
    return (tmp_4, in_5, in_6)

@triton.jit
def fused_muladd_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    n_heads,
    seq_len,
    feat_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    total_elements = batch_size * n_heads * seq_len * feat_dim
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices
    batch = offsets // (n_heads * seq_len * feat_dim)
    remainder = offsets % (n_heads * seq_len * feat_dim)
    head = remainder // (seq_len * feat_dim)
    remainder = remainder % (seq_len * feat_dim)
    seq = remainder // feat_dim
    feat = remainder % feat_dim
    
    # Load x, scale, and bias values
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Handle scale broadcasting: scale is [seq_len, feat_dim], needs to be broadcasted
    scale_offset = seq * feat_dim + feat
    scale_val = tl.load(scale_ptr + scale_offset, mask=scale_offset < scale_ptr.shape[0], other=1.0)
    
    bias_val = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: y = x * scale + bias
    out_val = x_val * scale_val + bias_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_multiply_add(x, bias, scale):
    batch_size, n_heads, seq_len, feat_dim = x.shape
    
    # Calculate optimal block size
    total_elements = batch_size * n_heads * seq_len * feat_dim
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_muladd_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=scale,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        n_heads=n_heads,
        seq_len=seq_len,
        feat_dim=feat_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_multiply_add