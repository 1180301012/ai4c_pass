import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Pattern to match:
    - Multiply in_2 * in_1 (with broadcasting)
    - Add in_0 (with broadcasting)  
    - Unbind along dim=2
    """
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_mul_add_unbind_kernel(
    in_0_ptr,  # [2, 128]
    in_1_ptr,  # [1, 1, 2, 128]
    in_2_ptr,  # [B, N, 1, 128]
    out_0_ptr,  # [B, N, D] - first slice
    out_1_ptr,  # [B, N, D] - second slice
    B, N, D,  # B=batch, N=17, D=128
):
    """
    Optimized kernel: each program processes one (B, N) pair across all D (128).
    Fixed BLOCK_D=128 for D=128 case.
    """
    # Get program ID - each program handles one (batch, row) pair
    pid = tl.program_id(0)
    
    # Compute batch and row indices
    b_idx = pid // N
    n_idx = pid % N
    
    # Check if this is a valid index
    if b_idx >= B:
        return
    
    # Create offset array for D dimension - fixed size 128
    d_offsets = tl.arange(0, 128)
    
    # Load from in_0: [2, D] - broadcasted
    in_0_val_0 = tl.load(in_0_ptr + 0 * D + d_offsets)
    in_0_val_1 = tl.load(in_0_ptr + 1 * D + d_offsets)
    
    # Load from in_1: [1, 1, 2, D] - broadcasted
    in_1_val_0 = tl.load(in_1_ptr + 0 * D + d_offsets)
    in_1_val_1 = tl.load(in_1_ptr + 1 * D + d_offsets)
    
    # Load from in_2: [B, N, 1, D]
    in_2_offset = b_idx * N * D + n_idx * D + d_offsets
    in_2_val = tl.load(in_2_ptr + in_2_offset)
    
    # Compute for both slices with FMA
    result_0 = in_2_val * in_1_val_0 + in_0_val_0
    result_1 = in_2_val * in_1_val_1 + in_0_val_1
    
    # Store outputs: [B, N, D]
    out_offset = b_idx * N * D + n_idx * D + d_offsets
    tl.store(out_0_ptr + out_offset, result_0)
    tl.store(out_1_ptr + out_offset, result_1)


@torch.fx.wrap
def fused_mul_add_unbind(in_0, in_1, in_2):
    """
    Fused operation combining multiply, add, unbind
    
    in_0: [2, 128]
    in_1: [1, 1, 2, 128]
    in_2: [B, 17, 1, 128]
    
    Returns:
    - Tuple of 2 tensors [B, 17, 128] each
    """
    # Get shapes
    B = in_2.shape[0]
    N = in_2.shape[1]  # Should be 17
    D = in_2.shape[3]  # Should be 128
    
    # Create output tensors
    out_0 = torch.empty((B, N, D), dtype=in_2.dtype, device=in_2.device)
    out_1 = torch.empty((B, N, D), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel - one program per (batch, row) pair
    grid = (B * N,)
    
    fused_mul_add_unbind_kernel[grid](
        in_0, in_1, in_2,
        out_0, out_1,
        B, N, D,
        num_warps=4,
    )
    
    return (out_0, out_1)


def replacement_func():
    return fused_mul_add_unbind