"""
Pass to fuse embedding lookup + permute + unsqueeze + expand + contiguous operations.
This pattern appears in relative attention bias computation in transformers.
"""
import torch
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
    """
    tmp_1 = in_1.to(device=torch.device('cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    """
    Extract arguments for the fused kernel.
    route_string differentiates different tensor shapes.
    """
    # Determine route based on in_0 shape
    if in_0.shape == torch.Size([32, 4]):
        route = "32x4"
    elif in_0.shape == torch.Size([32, 12]):
        route = "32x12"
    else:
        route = "default"
    
    # Determine the expand shape hint
    expand_batch = 1
    expand_heads = in_0.shape[0] if len(in_0.shape) > 0 else 32
    
    return (in_0, in_1, route, expand_batch, expand_heads)


@triton.jit
def fused_embedding_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    indices_stride_0,
    indices_stride_1,
    weight_stride_0,
    weight_stride_1,
    out_batch,
    out_heads,
    out_h,
    out_w,
    num_indices_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for embedding + permute + unsqueeze + expand + contiguous.
    
    Args:
        indices_ptr: pointer to indices tensor [H, W]
        weight_ptr: pointer to weight tensor [num_heads, head_dim] or [num_heads]
        output_ptr: pointer to output tensor [batch, num_heads, H, W]
        indices_stride_0, indices_stride_1: strides for indices
        weight_stride_0, weight_stride_1: strides for weight
        out_batch, out_heads, out_h, out_w: output dimensions
        num_indices_elements: total number of indices (H * W)
    """
    # Each program processes a block of output elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_indices_elements
    
    # Compute flat indices to H, W coordinates
    flat_indices = offsets
    h_idx = flat_indices // out_w
    w_idx = flat_indices % out_w
    
    # Load indices values
    idx_h = tl.load(indices_ptr + h_idx * indices_stride_0 + w_idx * indices_stride_1, mask=mask)
    
    # Compute output offsets for all heads (output shape: [batch, heads, H, W])
    # Each output element has batch=0, head from 0 to out_heads-1, and h, w position
    # We need to compute base offset for this position
    out_offset_base = h_idx * out_w + w_idx
    
    # Pre-allocate output accumulator
    output_values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over heads (usually small, like 32)
    for head_idx in range(out_heads):
        # Weight shape: [num_heads, head_dim] or [num_heads]
        # For embedding with shape [32, head_dim], idx_h selects from first dim
        # For embedding with shape [32], idx_h directly indexes
        
        if weight_stride_1 != 0:
            # 2D weight: [num_heads, head_dim]
            # Load weight for this head (all head_dim elements, but we take first)
            weight_offset = head_idx * weight_stride_0
            weight_val = tl.load(weight_ptr + weight_offset, mask=True)
        else:
            # 1D weight: [num_heads]
            weight_offset = head_idx * weight_stride_0
            weight_val = tl.load(weight_ptr + weight_offset, mask=True)
        
        # Embedding lookup: output[batch=0, head, h, w] = weight[head, indices[h, w]]
        # Or weight[indices[h, w], head] depending on layout
        
        # For the pattern: embedding table is [num_heads, head_dim]
        # indices are [H, W] with values in range [0, num_vocab)
        # output should be [H, W, num_heads] -> permuted to [num_heads, H, W]
        
        # The embedding output at position [h, w] is weight[indices[h, w]]
        # We permute to [num_heads, H, W] where num_heads dimension comes first
        
        # Actually, looking at the embedding: weight is [num_heads, head_dim]
        # Output of embedding is [H, W, num_heads * head_dim] = [H, W, 32]
        # Then permute to [32, H, W]
        
        # For simplicity, we'll handle the 1D weight case (weight shape [32])
        # where output at [head, h, w] = weight[indices[h, w], head]
        # But wait, that's not how embedding works...
        
        # Embedding: weight[num, :] selects the embedding vector
        # weight[indices[h,w]] gives the embedding vector
        # The pattern uses padding_idx=2.0 (sentinel value)
        
        # For weight shape [32, 4]:
        # - embedding(indices, weight) returns [H, W, 32] if indices is [H, W]
        # - Actually no: embedding looks up weight[indices] -> if weight is [32, 4], output is [H, W, 4]
        
        # Let me reconsider: in_0 shape is [32, 4] or [32, 12]
        # torch.nn.functional.embedding(indices, weight) where indices is [H, W]
        # Output is [H, W, 4] or [H, W, 12] (last dim matches weight's last dim)
        
        # Then permute [2, 0, 1] makes it [4, H, W] or [12, H, W]
        # Then unsqueeze to [1, 4, H, W] or [1, 12, H, W]
        # expand to [1, 4, H, W] or [1, 12, H, W] (first dim is 1, -1 means num_heads stays)
        
        # So output dim is [batch=1, num_heads, H, W] where num_heads = 4 or 12
        output_values = output_values  # Placeholder
    
    # Actually, we need to handle embedding correctly
    # weight is [num_heads, head_dim] (e.g., [32, 4])
    # embedding lookup: output[h, w, :] = weight[indices[h,w], :]
    # output shape is [H, W, num_heads*head_dim] but wait...
    
    # torch.nn.functional.embedding expects:
    # - weight: [num_embeddings, embedding_dim]
    # - indices: [..., N] (arbitrary leading dimensions)
    # - output: [..., N, embedding_dim]
    
    # So for weight [32, 4] and indices [H, W]:
    # output is [H, W, 4]
    
    # Then permute [2, 0, 1] -> [4, H, W]
    # Then unsqueeze -> [1, 4, H, W]
    
    # expand to [1, 4, H, W] (keeping the first -1 as 4)
    
    # So output is [1, 4, H, W] where H=W=45 (or 11 or 7)
    
    # The key is: output[0, k, h, w] = weight[indices[h, w], k] for k in [0, num_heads)
    
    # Re-implementing the kernel for this case:
    pass


# Let me write a simpler, correct implementation
@triton.jit
def fused_embedding_kernel_v2(
    indices_ptr,
    weight_ptr,
    output_ptr,
    indices_stride_0,
    indices_stride_1,
    weight_stride_0,
    weight_stride_1,
    out_batch,
    out_heads,
    out_h,
    out_w,
    num_indices_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for embedding + permute + unsqueeze + expand + contiguous.
    Handles embedding lookup with weight shape [num_heads, head_dim].
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_indices_elements
    
    # For output [batch, heads, H, W], we process each (h, w) position
    # and compute output values for all heads
    
    # Compute flat position to H, W
    flat_idx = offsets
    h_idx = flat_idx // out_w
    w_idx = flat_idx % out_w
    
    # Load the index value at (h, w)
    # indices has shape [H, W], so we compute proper offset
    idx_value = tl.load(
        indices_ptr + h_idx * indices_stride_0 + w_idx * indices_stride_1,
        mask=mask,
        other=0
    )
    
    # For embedding with weight [num_heads, head_dim]:
    # output[h, w, :] = weight[index_value, :]
    # After permute [2, 0, 1]: output[:][h][w] = original_output[h][w][:]
    
    # For the fused output [batch, heads, H, W]:
    # We need output[0, head, h, w] = weight[idx_value, head]
    # where head ranges from 0 to out_heads-1
    
    # For out_heads=4 or 12, we need to load from weight[idx_value, head]
    # weight layout: [num_heads, head_dim] but num_heads * head_dim = 32
    # For head_dim=4: num_heads=8 or similar
    # For head_dim=12: num_heads=8 or similar
    
    # Actually, looking at shape [32, 4], this means:
    # - 32 embeddings in the table
    # - 4 dimensions each
    # output[0, k, h, w] = weight[idx_value, k] where k in [0, 4)
    
    # Wait, let me reconsider the embedding behavior
    # torch.nn.functional.embedding(indices, weight) with weight [32, 4]
    # - indices: [..., N]
    # - output: [..., N, 4] where each output[..., i, :] = weight[indices[..., i]]
    
    # For indices shape [H, W] with weight [32, 4]:
    # - output shape is [H, W, 4]
    # - output[h, w, k] = weight[indices[h, w], k]
    
    # Then permute [2, 0, 1]: [4, H, W]
    # - permuted[h, k, w] = original[w, h, k]
    # Or in other words: permuted[k, h, w] = original[h, w, k]
    
    # Then unsqueeze: [1, 4, H, W]
    # Then expand: same shape [1, 4, H, W]
    
    # So output[0, k, h, w] = weight[indices[h, w], k]
    
    # For BLOCK_SIZE processing, we handle one (h,w) position per thread
    # and need to compute values for all out_heads
    
    # Create output pointer offsets for all heads
    # output is [batch, heads, H, W], contiguous
    # offset = batch * heads * H * W + head * H * W + h * W + w
    
    head_0_offset = out_h * out_w
    base_offset = h_idx * out_w + w_idx  # offset within a head slice
    
    result = tl.zeros((BLOCK_SIZE, out_heads), dtype=tl.float32) if out_heads <= 32 else None
    
    # Due to variable indexing complexity, we'll use a simpler per-element approach
    # defined in the final kernel below


# Final correct implementation
@triton.jit
def fused_embedding_final_kernel(
    indices_ptr,
    weight_ptr,
    output_ptr,
    indices_batch,
    indices_h,
    indices_w,
    indices_stride_h,
    indices_stride_w,
    weight_num,
    weight_dim,
    weight_stride_0,
    weight_stride_1,
    out_batch,
    out_heads,
    out_h,
    out_w,
    out_stride_batch,
    out_stride_heads,
    out_stride_h,
    out_stride_w,
    output_num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: to(device) + embedding + permute + unsqueeze + expand + contiguous.
    
    This kernel:
    1. Takes indices tensor [H, W] on CPU and weight [num_heads, head_dim] on GPU
    2. Performs embedding lookup: output[h, w, :] = weight[indices[h,w], :]
    3. Permutes to [heads, H, W]
    4. Adds batch dimension: [1, heads, H, W]
    5. Expands to [batch, heads, H, W] (batch is 1)
    6. Returns contiguous output [batch, heads, H, W]
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
    
    # Compute indices position (h_idx, w_idx)
    # indices[h_idx, w_idx] gives the lookup index
    idx_val = tl.load(
        indices_ptr + h_idx * indices_stride_h + w_idx * indices_stride_w,
        mask=mask,
        other=0.0
    )
    
    # For embedding output[0, head, h, w] = weight[indices[h,w], head]
    # weight has shape [weight_num, weight_dim]
    # but our output only has out_heads dimension
    
    # Actually for weight shape [32, 4] and out_heads=4:
    # output[0, k, h, w] = weight[indices[h,w], k] for k in [0, 4)
    
    # But weight has 32 entries, indices values range from 0 to 31
    # Then we select head dimension from weight
    
    # Compute weight base offset for this index
    # weight[idx_val, head_idx] = weight_ptr + idx_val * weight_stride_0 + head_idx * weight_stride_1
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
def fused_embedding_wrapper(weight, indices, route, expand_batch, expand_heads):
    """
    Wrapper function for the fused embedding + permute + unsqueeze + expand + contiguous.
    
    Args:
        weight: embedding table [num_heads, head_dim] on GPU
        indices: indices tensor [H, W] on CPU (will be moved to GPU)
        route: string to identify which kernel config to use
        expand_batch: batch dimension for expand (usually 1)
        expand_heads: number of heads (from weight shape)
    
    Returns:
        output tensor [expand_batch, expand_heads, H, W]
    """
    # Move indices to GPU if needed
    if indices.device.type == 'cpu':
        indices = indices.to(device='cuda')
    
    H, W = indices.shape
    num_heads = weight.shape[0]
    head_dim = weight.shape[1] if len(weight.shape) > 1 else 1
    
    # The output has shape [expand_batch, head_dim, H, W]
    # head_dim is extracted from weight's last dim, which is 4 or 12
    out_heads = head_dim
    out_batch = expand_batch
    output_num_elements = out_batch * out_heads * H * W
    
    # Allocate output
    output = torch.empty(
        (out_batch, out_heads, H, W),
        dtype=weight.dtype,
        device=weight.device
    )
    
    # Determine block size based on output size
    if output_num_elements <= 512:
        block_size = 64
    elif output_num_elements <= 4096:
        block_size = 256
    elif output_num_elements <= 32768:
        block_size = 512
    else:
        block_size = 1024
    
    num_programs = (output_num_elements + block_size - 1) // block_size
    
    # Grid configuration
    grid = (num_programs,)
    
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
    fused_embedding_final_kernel[grid](
        indices,
        weight,
        output,
        1,  # indices_batch (not directly used, but keeping for stride calc)
        H,
        W,
        indices_stride_h,
        indices_stride_w,
        weight.shape[0],
        weight.shape[1] if len(weight.shape) > 1 else 1,
        weight_stride_0,
        weight_stride_1,
        out_batch,
        out_heads,
        H,
        W,
        out_stride_batch,
        out_stride_heads,
        out_stride_h,
        out_stride_w,
        output_num_elements,
        BLOCK_SIZE=block_size,
    )
    
    return output


def replacement_func():
    """
    Returns the replacement function for the fused operation.
    """
    return fused_embedding_wrapper