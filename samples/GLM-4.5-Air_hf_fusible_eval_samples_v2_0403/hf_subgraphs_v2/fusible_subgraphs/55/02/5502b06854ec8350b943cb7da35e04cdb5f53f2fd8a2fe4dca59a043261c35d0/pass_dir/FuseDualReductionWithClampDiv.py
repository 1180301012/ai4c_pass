import torch
import triton
import triton.language as tl

# Pattern matching for the entire computation: dual reduction + clamp + division
def pattern(a, b):
    # tmp_0 = a.to(torch.float32)
    tmp_0 = a.to(torch.float32)
    
    # tmp_1 = b * tmp_0  
    tmp_1 = b * tmp_0
    
    # tmp_2 = torch.sum(tmp_1, 1)
    tmp_2 = torch.sum(tmp_1, 1)
    
    # tmp_3 = tmp_0.sum(1)
    tmp_3 = tmp_0.sum(1)
    
    # tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    
    # tmp_5 = tmp_2 / tmp_4
    tmp_5 = tmp_2 / tmp_4
    
    # tmp_6 = torch.cat([tmp_5], 1)
    tmp_6 = torch.cat([tmp_5], 1)
    
    return tmp_6

@triton.jit
def fused_reduction_kernel(
    a_ptr, b_ptr, 
    out_ptr,
    n_batch, n_seq, n_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for each (batch, hidden) pair
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # hidden
    
    # Early return for invalid indices
    if pid_m >= n_batch or pid_n >= n_hidden:
        return
    
    # Load sequence data for this (batch, hidden) pair
    # For small sequence sizes, sequential access is efficient
    sum_a = 0.0
    sum_product = 0.0
    
    for seq_idx in range(n_seq):
        # Calculate memory offset
        offset = pid_m * n_seq * n_hidden + seq_idx * n_hidden + pid_n
        
        # Load elements
        a_val = tl.load(a_ptr + offset)
        b_val = tl.load(b_ptr + offset)
        
        # Accumulate sums
        sum_a += a_val
        sum_product += b_val * a_val
    
    # Clamp and compute division
    clamped_sum_a = tl.maximum(sum_a, 1e-09)
    result = sum_product / clamped_sum_a
    
    # Store result
    output_offset = pid_m * n_hidden + pid_n
    tl.store(out_ptr + output_offset, result)

@torch.fx.wrap
def fused_reduction_computation(a, b):
    """Fused computation that reduces multiple operations into a single kernel"""
    # Convert a to float32 once (no-op if already float32)
    a_float32 = a.to(torch.float32)
    
    if a_float32.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {a_float32.dim()}D")
    
    n_batch, n_seq, n_hidden = a_float32.shape
    
    # Create output tensor [batch, hidden]
    out = torch.empty((n_batch, n_hidden), dtype=torch.float32, device=a.device)
    
    # Launch with 2D grid: each program handles one (batch, hidden) pair
    # For small problem sizes, this simple approach is often most efficient
    grid = (n_batch, n_hidden)
    
    fused_reduction_kernel[grid](
        a_ptr=a_float32,
        b_ptr=b,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq, 
        n_hidden=n_hidden,
        BLOCK_SIZE=128  # Not used in this kernel, but required for signature
    )
    
    # Add sequence dimension back and concatenate (no-op)
    result = out.unsqueeze(1)
    
    return result

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    return fused_reduction_computation