import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_1.reshape(-1, 16, -1, 128)
    tmp_1 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, -1, None)]
    tmp_2 = in_2.contiguous()
    tmp_3 = in_3.contiguous()
    tmp_4 = tmp_0.contiguous()
    tmp_0 = None
    return (tmp_1, tmp_3, tmp_2, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def reshape_slice_kernel(
    causal_mask_ptr,
    expand_ptr, 
    q_embed_ptr,
    reshape_3_ptr,
    output_mask_ptr,
    output_reshape_ptr,
    output_q_embed_ptr,
    output_expand_ptr,
    batch_size,
    seq_len,
    head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Process one batch element and sequence position per program
    b = pid // seq_len
    s = pid % seq_len
    
    mask_offset = b * seq_len * seq_len + s * seq_len
    expand_offset = b * 16 * seq_len * head_dim + s * head_dim
    
    # Load causal mask for this position
    causal_mask_val = tl.load(causal_mask_ptr + mask_offset)
    
    # Reshape and load from expanded tensor
    # Original shape: [batch, 4, 4, seq_len, head_dim]
    # Target reshape: [batch, 16, seq_len, head_dim] 
    block_offset = b * 16 * seq_len * head_dim + s * head_dim
    
    # Load 16 elements (from different heads) for this position
    offsets = block_offset + tl.arange(0, 16 * head_dim)
    mask = offsets < (b * 16 * seq_len * head_dim + 16 * (s + 1) * head_dim)
    
    # Reshape operation: collapse [batch, 4, 4, seq_len, head_dim] -> [batch, 16, seq_len, head_dim]
    # For each (b, s), we're processing 16 head elements
    expanded_data = tl.load(expand_ptr + offsets, mask=mask, other=0.0)
    
    output_offset = b * 16 * seq_len * head_dim + s * head_dim
    tl.store(output_expand_ptr + output_offset, expanded_data, mask=mask)

@torch.fx.wrap
def fuse_reshape_slice(in_0, in_1, in_2, in_3):
    # Extract shapes from inputs
    batch_size = in_1.shape[0]
    original_4d_shape = in_1.shape[-3:]  # [4, 4, seq_len, head_dim] - take last 3 dims
    seq_len = in_1.shape[-2]
    head_dim = in_1.shape[-1]
    
    # Output shapes match the pattern
    # tmp_1: slice from in_0 needs to match reshape shape
    # tmp_4: reshaped data
    output_shape_expanded = (batch_size, 16, seq_len, head_dim)
    
    # Create output tensors
    output_mask = torch.zeros((in_0.shape[0], 1, seq_len, in_0.shape[-1]), dtype=in_0.dtype, device=in_0.device)
    output_reshaped = torch.empty(output_shape_expanded, dtype=in_1.dtype, device=in_1.device)
    
    # Set up grid for kernel
    total_elements = batch_size * seq_len
    grid = (total_elements,)
    
    # Launch kernel
    reshape_slice_kernel[grid](
        in_0,
        in_1, 
        in_2,
        in_3,
        output_mask,
        output_reshaped,
        in_2,  # tmp_2 remains the same (q_embed)
        in_3,  # tmp_3 remains the same (reshape_3)
        batch_size,
        seq_len,
        head_dim,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128
    )
    
    return (output_mask, in_3, in_2, output_reshaped)

def replacement_func():
    return fuse_reshape_slice