import torch
import triton
import triton.language as tl

def pattern(tmp_10):
    tmp_11 = tmp_10[(Ellipsis, slice(None, 256, None))]
    tmp_12 = tmp_10[(Ellipsis, slice(-256, None, None))]
    return tmp_11, tmp_12

def replacement_args(tmp_10):
    return (tmp_10,)

@triton.jit
def parallel_slice_kernel(
    input_ptr,
    out1_ptr, out2_ptr,
    M_total,  # Total dimensions before the last dimension
    N_total,   # Original last dimension size (512)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    SLICE_SIZE: tl.constexpr,
):
    # Program Ids
    pid_M = tl.program_id(0)
    pid_N1 = tl.program_id(1)  # For first slice (first 256)
    pid_N2 = tl.program_id(2)  # For second slice (last 256)
    
    # Ranges
    m_offsets = pid_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n1_offsets = pid_N1 * SLICE_SIZE + tl.arange(0, SLICE_SIZE)  # First 256
    n2_offsets = (N_total - SLICE_SIZE) + (pid_N2 * SLICE_SIZE + tl.arange(0, SLICE_SIZE))  # Last 256
    
    # Create masks
    mask_M = m_offsets < M_total
    mask_N1 = n1_offsets < SLICE_SIZE
    mask_N2 = n2_offsets < N_total
    
    # Process first slice (first 256 elements)
    if pid_N1 == 0:  # Only one program needed for the slice
        # Load data for first slice
        input_idx1 = m_offsets[:, None] * N_total + n1_offsets[None, :]
        mask1 = mask_M[:, None] & mask_N1[None, :]
        data1 = tl.load(input_ptr + input_idx1, mask=mask1, other=0.0)
        
        # Store to first output
        out_idx1 = m_offsets[:, None] * SLICE_SIZE + n1_offsets[None, :]
        tl.store(out1_ptr + out_idx1, data1, mask=mask1)
    
    # Process second slice (last 256 elements)
    if pid_N2 == 0:  # Only one program needed for the slice
        # Load data for second slice
        input_idx2 = m_offsets[:, None] * N_total + n2_offsets[None, :]
        mask2 = mask_M[:, None] & (n2_offsets < N_total)[None, :]
        data2 = tl.load(input_ptr + input_idx2, mask=mask2, other=0.0)
        
        # Store to second output (normalized indices within slice)
        out_idx2 = m_offsets[:, None] * SLICE_SIZE + (n2_offsets - (N_total - SLICE_SIZE))[None, :]
        tl.store(out2_ptr + out_idx2, data2, mask=mask2)

@torch.fx.wrap
def fused_parallel_slices(tmp_10):
    # Input tmp_10 has shape [300, 1, 512] (from the linear operation)
    # We need to slice both on the last dimension: first 256 and last 256
    M_total = tmp_10.numel() // 512  # Total elements / last dimension size
    N_total = 512
    SLICE_SIZE = 256
    
    # Output shapes will be [M_total, SLICE_SIZE] for each slice
    out1_shape = (M_total, SLICE_SIZE)
    out2_shape = (M_total, SLICE_SIZE)
    
    out1 = torch.empty(out1_shape, dtype=tmp_10.dtype, device=tmp_10.device)
    out2 = torch.empty(out2_shape, dtype=tmp_10.dtype, device=tmp_10.device)
    
    # Number of programs
    def cdiv(a, b):
        return (a + b - 1) // b
    
    GRID_M = cdiv(M_total, 128)   # Process M dimension in chunks of 128
    GRID_SLICE = 1  # Each slice processed by one program
    
    # Launch kernel for parallel slicing
    parallel_slice_kernel[(GRID_M, GRID_SLICE, GRID_SLICE)](
        tmp_10,
        out1, out2,
        M_total, N_total,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=SLICE_SIZE,
        SLICE_SIZE=SLICE_SIZE,
    )
    
    return out1, out2

def replacement_func():
    return fused_parallel_slices