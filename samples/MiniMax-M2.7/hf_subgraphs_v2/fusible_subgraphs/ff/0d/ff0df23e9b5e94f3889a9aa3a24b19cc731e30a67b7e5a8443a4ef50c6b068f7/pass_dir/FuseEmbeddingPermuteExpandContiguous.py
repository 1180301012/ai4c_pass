"""
Pass to fuse embedding lookup + permute + unsqueeze + expand + contiguous operations.
This pattern appears in relative attention bias computation in transformers.
"""
import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern:
    1. Move indices to GPU
    2. Embedding lookup
    3. Permute [2, 0, 1]
    4. Unsqueeze(0)
    5. Expand to (batch, -1, H, W)
    6. Contiguous
    
    Args:
        in_0: weight tensor [32, head_dim] on GPU
        in_1: indices tensor [H, W] on CPU
    """
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    """
    Extract arguments for the fused kernel.
    
    The embedding weight is [num_entries, head_dim] = [32, 4] or [32, 12].
    The indices are [H, W] with values in [0, 31].
    The output should be [1, head_dim, H, W].
    """
    # Determine the head_dim from weight shape
    # Weight shapes are [32, 4] or [32, 12] - always 2D with head_dim in dim 1
    head_dim = in_0.shape[1]
    
    # The expand shape batch dimension is always 1
    expand_batch = 1
    
    return (in_0, in_1, head_dim, expand_batch)


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
    
    This is equivalent to:
    1. embedding_lookup = torch.nn.functional.embedding(indices, weight) -> [H, W, head_dim]
    2. permuted = embedding_lookup.permute([2, 0, 1]) -> [head_dim, H, W]
    3. unsqueezed = permuted.unsqueeze(0) -> [1, head_dim, H, W]
    4. expanded = unsqueezed.expand((1, -1, H, W)) -> [1, head_dim, H, W]
    5. contiguous() -> [1, head_dim, H, W]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_num_elements
    
    # Decode output position: [batch, head, h, w]
    # Output layout (contiguous): batch * (heads*H*W) + head * (H*W) + h * W + w
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
    # weight has shape [weight_entries, weight_dim]
    # We access weight[idx_val, head_idx]
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
    
    Args:
        weight: embedding table [32, head_dim] on GPU, dtype float16/bfloat16/float32
        indices: indices tensor [H, W] on CPU, dtype int64
        head_dim: dimension of each embedding (4 or 12)
        expand_batch: batch dimension for expand (usually 1)
    
    Returns:
        output tensor [expand_batch, head_dim, H, W]
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
    weight_stride_1 = weight.stride(1) if len(weight.shape) > 1 else 0
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


def replacement_func():
    """
    Returns the replacement function for the fused operation.
    """
    from pass_dir.shared_kernel import fused_embedding_dispatch
    return fused_embedding_dispatch