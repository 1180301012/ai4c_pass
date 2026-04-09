import torch
import triton
import triton.language as tl



def pattern(base_tensor):
    """
    Optimized pattern that matches the attention mask computation.
    """
    # tmp_10 = base_tensor.unsqueeze(2)  # [1, 361, 1, 49]
    tmp_10 = base_tensor.unsqueeze(2)
    
    # tmp_11 = base_tensor.unsqueeze(3)  # [1, 361, 49, 1]  
    tmp_11 = base_tensor.unsqueeze(3)
    
    # tmp_12 = tmp_10 - tmp_11           # [1, 361, 49, 49]
    tmp_12 = tmp_10 - tmp_11
    
    # tmp_13 = tmp_12 != 0
    tmp_13 = tmp_12 != 0
    
    # tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    
    # tmp_15 = tmp_12 == 0
    tmp_15 = tmp_12 == 0
    
    # tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    
    # Return the final result that's observable outside (tmp_16)
    return tmp_16

def replacement_args(base_tensor):
    return (base_tensor,)

@triton.jit
def optimized_attention_mask_kernel(
    base_ptr,
    out_ptr,
    n_head: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for attention mask computation using 1D grid for better performance.
    """
    pid = tl.program_id(0)
    
    # Get offsets for the entire block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    total_elements = n_head * seq_len * seq_len
    mask = offsets < total_elements
    
    # Calculate head indices and remaining indices
    head_idx = offsets // (seq_len * seq_len)
    remaining = offsets % (seq_len * seq_len)
    
    # Use memory coalescing-friendly access patterns
    # Process contiguous memory accesses together
    
    # Optimized memory access: read base_tensor in efficient patterns
    # For each head, we need base_tensor[head_idx, :, 0] and base_tensor[head_idx, 0, :]
    # But we'll compute this on the fly for memory efficiency
    
    # For each position (head_idx, j, k), compute:
    # val1 = base_tensor[head_idx, j, 0] 
    # val2 = base_tensor[head_idx, 0, k]
    
    # Vectorized computation for better throughput
    j_idx = (remaining // seq_len) % seq_len
    k_idx = remaining % seq_len
    
    # Position 1: base_tensor[head_idx, j_idx, 0]
    # Position 2: base_tensor[head_idx, 0, k_idx]
    pos1 = head_idx * seq_len + j_idx
    pos2 = head_idx * seq_len + k_idx
    
    # Use efficient memory access with optimized masks
    mask_safe = mask & (head_idx < n_head) & (j_idx < seq_len) & (k_idx < seq_len)
    
    # Optimized loads with better memory locality
    val1 = tl.load(base_ptr + pos1, mask=mask_safe, other=0.0)
    val2 = tl.load(base_ptr + pos2, mask=mask_safe, other=0.0)
    
    # Compute attention mask with vectorized operations
    diff = val1 - val2
    result = tl.where(diff != 0, -1000.0, 0.0)
    
    # Coalesced memory store
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_fused_attention_mask(base_tensor):
    """
    Optimized wrapper function with adaptive block size selection.
    """
    # Get tensor shape - expected: [1, 361, 49]
    assert len(base_tensor.shape) == 3, "expected 3D tensor"
    batch_size, n_head, seq_len = base_tensor.shape
    
    # Expected shape: [1, 361, 49] 
    assert batch_size == 1, "only batch size 1 is supported"
    assert seq_len == 49, f"expected seq_len=49, got {seq_len}"
    
    # Create output tensor [1, 361, 49, 49] with same dtype as input
    out_shape = (batch_size, n_head, seq_len, seq_len)
    out = torch.empty(out_shape, dtype=base_tensor.dtype, device=base_tensor.device)
    
    # Adaptive block size based on workload
    total_elements = n_head * seq_len * seq_len
    
    if total_elements > 0:
        # Try different block sizes for optimal performance
        if total_elements < 16384:
            BLOCK_SIZE = 256   # Small workloads
        elif total_elements < 65536:
            BLOCK_SIZE = 512   # Medium workloads  
        else:
            BLOCK_SIZE = 1024  # Large workloads
        
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel with optimized configuration
        optimized_attention_mask_kernel[(num_programs,)](
            base_ptr=base_tensor,
            out_ptr=out.reshape(-1),  # Flatten for 1D processing
            n_head=n_head,
            seq_len=seq_len,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return optimized_fused_attention_mask