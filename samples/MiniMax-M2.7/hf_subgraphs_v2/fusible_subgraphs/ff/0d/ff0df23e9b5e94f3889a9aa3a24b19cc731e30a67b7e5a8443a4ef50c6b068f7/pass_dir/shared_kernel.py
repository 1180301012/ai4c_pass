"""
Shared kernel functions for embedding fusion passes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    # Indices tensor info
    indices_h,
    indices_w,
    indices_stride_h,
    indices_stride_w,
    # Weight tensor info
    weight_entries,
    weight_dim,
    weight_stride_0,
    weight_stride_1,
    # Output tensor info
    out_batch,
    out_heads,
    out_h,
    out_w,
    out_stride_batch,
    out_stride_heads,
    out_stride_h,
    out_stride_w,
    # Total elements
    output_num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: to(device) + embedding + permute + unsqueeze + expand + contiguous.
    
    Computes: output[0, head, h, w] = weight[indices[h, w], head]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_num_elements
    
    # Decode output position: [batch, head, h, w]
    batch_idx = offsets // (out_heads * out_h * out_w)
    remainder = offsets % (out_heads * out_h * out_w)
    head_idx = remainder // (out_h * out_w)
    remainder2 = remainder % (out_h * out_w)
    h_idx = remainder2 // out_w
    w_idx = remainder2 % out_w
    
    # Load indices value at (h_idx, w_idx)
    idx_val = tl.load(
        indices_ptr + h_idx * indices_stride_h + w_idx * indices_stride_w,
        mask=mask,
        other=0.0
    )
    
    # Embedding lookup: output[0, head, h, w] = weight[indices[h,w], head]
    weight_offset = idx_val * weight_stride_0 + head_idx * weight_stride_1
    result = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
    
    # Store result to output
    out_offset = (
        batch_idx * out_stride_batch +
        head_idx * out_stride_heads +
        h_idx * out_stride_h +
        w_idx * out_stride_w
    )
    tl.store(output_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_embedding_dispatch(weight, indices, head_dim, expand_batch):
    """
    Wrapper function for the fused embedding + permute + unsqueeze + expand + contiguous.
    """
    # Move indices to GPU
    if indices.device.type == 'cpu':
        indices = indices.to(device='cuda', non_blocking=True)
    
    H, W = indices.shape
    
    # Output shape: [batch, head_dim, H, W]
    out_batch = expand_batch
    out_heads = head_dim
    output_num_elements = out_batch * out_heads * H * W
    
    # Allocate output with same dtype as weight
    output = torch.empty(
        (out_batch, out_heads, H, W),
        dtype=weight.dtype,
        device=weight.device
    )
    
    # Select block size based on output size
    if output_num_elements <= 256:
        block_size = 32
    elif output_num_elements <= 2048:
        block_size = 128
    elif output_num_elements <= 16384:
        block_size = 256
    else:
        block_size = 512
    
    num_programs = (output_num_elements + block_size - 1) // block_size
    
    # Get strides
    indices_stride_h = indices.stride(0)
    indices_stride_w = indices.stride(1)
    weight_stride_0 = weight.stride(0)
    weight_stride_1 = weight.stride(1)
    out_stride_batch = output.stride(0)
    out_stride_heads = output.stride(1)
    out_stride_h = output.stride(2)
    out_stride_w = output.stride(3)
    
    # Launch kernel
    grid = (num_programs,)
    fused_embedding_kernel[grid](
        indices,
        weight,
        output,
        # Indices
        H, W, indices_stride_h, indices_stride_w,
        # Weight
        weight.shape[0], weight.shape[1],
        weight_stride_0, weight_stride_1,
        # Output
        out_batch, out_heads, H, W,
        out_stride_batch, out_stride_heads, out_stride_h, out_stride_w,
        # Total
        output_num_elements,
        BLOCK_SIZE=block_size,
    )
    
    return output