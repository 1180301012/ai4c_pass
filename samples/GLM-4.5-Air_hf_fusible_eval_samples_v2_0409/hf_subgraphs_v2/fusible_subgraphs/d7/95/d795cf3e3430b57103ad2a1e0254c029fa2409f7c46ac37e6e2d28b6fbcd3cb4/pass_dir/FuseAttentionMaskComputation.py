import torch
import triton
import triton.language as tl

def pattern(base_tensor):
    """
    Pattern that matches the attention mask computation:
    - base_tensor (tmp_9) is [1, 361, 49]
    - We reconstruct the entire computation sequence that produces tmp_16
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
def attention_mask_kernel(
    base_ptr,
    out_ptr,
    n_head: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for attention mask computation.
    
    This kernel computes the equivalent of:
    - unsq_2 = base_tensor.unsqueeze(2)  # [1, 361, 1, 49]  
    - unsq_3 = base_tensor.unsqueeze(3)  # [1, 361, 49, 1]
    - diff = unsq_2 - unsq_3             # [1, 361, 49, 49]
    - result = where(diff != 0, -1000.0, 0.0)
    
    The key insight: diff[i, head_i, j, k] = base[head_i, j, 0] - base[head_i, 0, k]
    """
    # We need 3 program IDs for: head, row, column
    pid_head = tl.program_id(0)
    pid_j = tl.program_id(1)  # row index in output [0, 49)
    pid_k = tl.program_id(2)  # column index in output [0, 49)
    
    # Check bounds (avoid chained boolean operators)
    if pid_head >= n_head:
        return
    if pid_j >= seq_len:
        return
    if pid_k >= seq_len:
        return
    
    # Load base tensor values at specific positions
    # base[head_i, j, 0] for the first term
    idx_base_j0 = pid_head * seq_len + pid_j
    val_base_j0 = tl.load(base_ptr + idx_base_j0)
    
    # base[head_i, 0, k] for the second term  
    idx_base_0k = pid_head * seq_len + pid_k
    val_base_0k = tl.load(base_ptr + idx_base_0k)
    
    # Compute difference
    diff = val_base_j0 - val_base_0k
    
    # Create attention mask value
    result = tl.where(diff != 0, -1000.0, 0.0)
    
    # Store result at position [1, pid_head, pid_j, pid_k]
    output_idx = pid_head * seq_len * seq_len + pid_j * seq_len + pid_k
    tl.store(out_ptr + output_idx, result)

@torch.fx.wrap
def fused_attention_mask(base_tensor):
    """
    Wrapper function to launch the Triton kernel for attention mask computation.
    
    This function computes the equivalent of:
    - tmp_10 = base_tensor.unsqueeze(2)  # [1, 361, 1, 49]
    - tmp_11 = base_tensor.unsqueeze(3)  # [1, 361, 49, 1] 
    - tmp_12 = tmp_10 - tmp_11           # [1, 361, 49, 49]
    - tmp_13 = tmp_12 != 0
    - tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    - tmp_15 = tmp_12 == 0
    - tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
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
    
    # Launch Triton kernel with 3D grid
    if n_head > 0 and seq_len > 0:
        # Grid dimensions: (n_head, seq_len, seq_len)
        attention_mask_kernel[(n_head, seq_len, seq_len)](
            base_ptr=base_tensor,
            out_ptr=out,
            n_head=n_head,
            seq_len=seq_len,
            BLOCK_SIZE=0,  # Not used for this 3D grid kernel
        )
    
    return out

def replacement_func():
    return fused_attention_mask