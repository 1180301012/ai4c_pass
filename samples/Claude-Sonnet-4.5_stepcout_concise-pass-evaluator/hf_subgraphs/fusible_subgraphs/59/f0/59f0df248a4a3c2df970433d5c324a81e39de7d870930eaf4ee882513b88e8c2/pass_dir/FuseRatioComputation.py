import torch
import triton
import triton.language as tl

def pattern(attention_mask, input_ids):
    """
    Pattern: Compute ratio of padding tokens and expand dimensions
    """
    tmp_7 = attention_mask.sum(-1)
    tmp_8 = input_ids.__eq__(2)
    tmp_9 = tmp_8.sum(-1)
    tmp_10 = tmp_9.float()
    tmp_11 = tmp_10 / tmp_7
    tmp_13 = 1 - tmp_11
    tmp_14 = tmp_13[slice(None, None, None), None, None]
    return tmp_14

def replacement_args(attention_mask, input_ids):
    return (attention_mask, input_ids)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_ratio_kernel(
    attention_mask_ptr,
    input_ids_ptr,
    output_ptr,
    batch_size,
    seq_len,
    stride_mask_batch,
    stride_mask_seq,
    stride_ids_batch,
    stride_ids_seq,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Offsets for this batch
    offs_seq = tl.arange(0, BLOCK_SIZE)
    mask = offs_seq < seq_len
    
    # Load attention_mask and input_ids for this batch
    mask_ptrs = attention_mask_ptr + batch_idx * stride_mask_batch + offs_seq * stride_mask_seq
    ids_ptrs = input_ids_ptr + batch_idx * stride_ids_batch + offs_seq * stride_ids_seq
    
    attention_vals = tl.load(mask_ptrs, mask=mask, other=0)
    input_vals = tl.load(ids_ptrs, mask=mask, other=0)
    
    # Sum attention_mask
    attention_sum = tl.sum(attention_vals.to(tl.float32), axis=0)
    
    # Count padding tokens (id == 2)
    is_padding = (input_vals == 2)
    padding_count = tl.sum(is_padding.to(tl.float32), axis=0)
    
    # Compute ratio and final value
    ratio = padding_count / attention_sum
    result = 1.0 - ratio
    
    # Store result
    output_idx = batch_idx
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def fused_ratio_computation(attention_mask, input_ids):
    # Get shapes
    batch_size = attention_mask.shape[0]
    seq_len = attention_mask.shape[1]
    
    # Output tensor
    output = torch.empty((batch_size, 1, 1), 
                         dtype=torch.float32, 
                         device=attention_mask.device)
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    # Launch kernel
    fused_ratio_kernel[grid](
        attention_mask,
        input_ids,
        output.view(batch_size),
        batch_size,
        seq_len,
        attention_mask.stride(0),
        attention_mask.stride(1),
        input_ids.stride(0),
        input_ids.stride(1),
    )
    
    return output

def replacement_func():
    return fused_ratio_computation