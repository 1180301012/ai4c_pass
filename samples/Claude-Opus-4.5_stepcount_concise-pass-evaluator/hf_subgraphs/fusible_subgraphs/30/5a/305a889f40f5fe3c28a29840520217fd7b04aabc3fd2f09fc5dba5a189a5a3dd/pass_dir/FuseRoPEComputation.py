import torch
import triton
import triton.language as tl

# Pattern for the mul + add + cat + type_as computation that's common to all models
def pattern(reshaped, mul_base, sin_emb, prefix, ref_tensor):
    """
    Matches: reshaped * sin_emb + mul_base, then cat with prefix, then type_as
    """
    tmp_5 = reshaped * sin_emb
    tmp_6 = mul_base + tmp_5
    tmp_7 = torch.cat([prefix, tmp_6], dim=2)
    tmp_8 = tmp_7.type_as(ref_tensor)
    return tmp_8


def replacement_args(reshaped, mul_base, sin_emb, prefix, ref_tensor):
    return (reshaped, mul_base, sin_emb, prefix, ref_tensor)


@triton.jit
def fused_mul_add_kernel_with_offset(
    reshaped_ptr, mul_base_ptr, sin_emb_ptr, output_ptr,
    n_elements,
    seq_len, head_dim, seq_len_plus_one,
    sin_stride_s, sin_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for mul + add writing to output with proper offset for concat"""
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values (contiguous access for reshaped, mul_base)
    reshaped_vals = tl.load(reshaped_ptr + offsets, mask=mask)
    mul_base_vals = tl.load(mul_base_ptr + offsets, mask=mask)
    
    # Compute sin_emb index
    d = offsets % head_dim
    local_seq = (offsets // head_dim) % seq_len
    sin_emb_idx = local_seq * sin_stride_s + d * sin_stride_d
    sin_emb_vals = tl.load(sin_emb_ptr + sin_emb_idx, mask=mask)
    
    # Compute: mul_base + reshaped * sin_emb
    result = mul_base_vals + reshaped_vals * sin_emb_vals
    
    # Compute output index: need to insert after prefix (at position 1 onwards)
    # Input flat index: batch_head_idx * seq_len * head_dim + local_seq * head_dim + d
    # Output flat index: batch_head_idx * (seq_len+1) * head_dim + (local_seq+1) * head_dim + d
    batch_head_seq = offsets // head_dim  # = batch_head_idx * seq_len + local_seq
    batch_head_idx = batch_head_seq // seq_len
    
    out_idx = batch_head_idx * seq_len_plus_one * head_dim + (local_seq + 1) * head_dim + d
    
    # Store
    tl.store(output_ptr + out_idx, result, mask=mask)


@triton.jit
def copy_prefix_kernel(
    prefix_ptr, output_ptr,
    n_batch_heads, head_dim, seq_len_plus_one,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy prefix to position 0 in output"""
    pid = tl.program_id(0)
    
    total = n_batch_heads * head_dim
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    d = offsets % head_dim
    bh = offsets // head_dim
    
    # Load prefix
    prefix_idx = bh * head_dim + d
    vals = tl.load(prefix_ptr + prefix_idx, mask=mask)
    
    # Store to output at seq=0
    out_idx = bh * seq_len_plus_one * head_dim + d
    tl.store(output_ptr + out_idx, vals, mask=mask)


@torch.fx.wrap
def fused_mul_add_cat(reshaped, mul_base, sin_emb, prefix, ref_tensor):
    """
    Fused computation: (reshaped * sin_emb + mul_base) concat with prefix, then type_as
    """
    batch, heads, seq_len, head_dim = reshaped.shape
    seq_len_plus_one = seq_len + 1
    
    # Output shape: [batch, heads, seq_len+1, head_dim]
    output = torch.empty(batch, heads, seq_len_plus_one, head_dim, 
                        dtype=ref_tensor.dtype, device=reshaped.device)
    
    n_batch_heads = batch * heads
    n_elements = n_batch_heads * seq_len * head_dim
    
    # Make input tensors contiguous and flatten
    reshaped_flat = reshaped.contiguous().view(-1)
    mul_base_flat = mul_base.contiguous().view(-1)
    prefix_flat = prefix.contiguous().view(-1)
    output_flat = output.view(-1)
    
    # Get sin_emb strides
    sin_stride_s = sin_emb.stride(0)
    sin_stride_d = sin_emb.stride(1)
    
    BLOCK_SIZE = 1024
    
    # Copy prefix
    total_prefix = n_batch_heads * head_dim
    grid_prefix = (triton.cdiv(total_prefix, BLOCK_SIZE),)
    copy_prefix_kernel[grid_prefix](
        prefix_flat, output_flat,
        n_batch_heads, head_dim, seq_len_plus_one,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Main computation
    grid_main = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_mul_add_kernel_with_offset[grid_main](
        reshaped_flat, mul_base_flat, sin_emb, output_flat,
        n_elements,
        seq_len, head_dim, seq_len_plus_one,
        sin_stride_s, sin_stride_d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_mul_add_cat