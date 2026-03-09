import torch
import triton
import triton.language as tl

# Pattern matching for the second branch: slicing, multiplication, and chunking
def pattern(in_0, in_1, in_3, X):
    tmp_5 = in_0[slice(None, None, None), slice(None, None, None), slice(None, X, None), slice(None, None, None)]
    tmp_6 = in_1[slice(None, None, None), slice(None, None, None), slice(None, X, None), slice(None, None, None)]
    tmp_7 = in_3 * tmp_5
    tmp_8 = in_3.chunk(2, dim=-1)
    tmp_9 = tmp_8[0]
    tmp_10 = tmp_8[1]
    return tmp_6, tmp_7, tmp_9, tmp_10

# Argument extraction function
def replacement_args(in_0, in_1, in_3, X):
    return (in_0, in_1, in_3, X)

# Optimized kernel for fused slicing, multiplication, and chunking
@triton.jit
def fused_slicing_chunk_kernel(
    in_0_ptr, 
    in_1_ptr, 
    in_3_ptr,
    out_6_ptr,  # sliced in_1
    out_7_ptr,  # in_3 * sliced in_0  
    out_9_ptr,  # first chunk of in_3
    out_10_ptr, # second chunk of in_3
    n0, n1, n2_slice, n3,
    n2_in3, n3_half,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one tile
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Compute memory offsets for sliced data (first n2_slice elements)
    off_in_0_slice = m * BLOCK_SIZE_M * n2_slice * n3 + n * BLOCK_SIZE_N
    off_in_1_slice = m * BLOCK_SIZE_M * n2_slice * n3 + n * BLOCK_SIZE_N
    off_in_3_slice = m * BLOCK_SIZE_M * n2_slice * n3 + n * BLOCK_SIZE_N
    off_out_6 = m * BLOCK_SIZE_M * n2_slice * n3 + n * BLOCK_SIZE_N
    off_out_7 = off_out_6
    
    # Create masks for sliced data
    slice_elements = n0 * n1 * n2_slice * n3
    mask_slice = off_in_0_slice < slice_elements
    
    # Load sliced input data  
    in_0_slice = tl.load(in_0_ptr + off_in_0_slice, mask=mask_slice, other=0.0)
    in_1_slice = tl.load(in_1_ptr + off_in_1_slice, mask=mask_slice, other=0.0)
    in_3_slice = tl.load(in_3_ptr + off_in_3_slice, mask=mask_slice, other=0.0)
    
    # Store slice results (tmp_6 = sliced in_1)
    tl.store(out_6_ptr + off_out_6, in_1_slice, mask=mask_slice)
    
    # Compute multiplication (tmp_7 = in_3 * tmp_5)
    tmp_7 = in_3_slice * in_0_slice
    tl.store(out_7_ptr + off_out_7, tmp_7, mask=mask_slice)
    
    # For chunking operations, handle separately due to different indexing
    # First process half of the chunks
    if n < n3_half:
        # Offsets for first chunk (first half of last dimension)
        off_in_3_first = m * BLOCK_SIZE_M * n2_in3 * n3 + n * BLOCK_SIZE_N
        off_out_9 = off_in_3_first
        
        mask_first = off_in_3_first < n0 * n1 * n2_in3 * n3_half
        if mask_first:
            # For the first chunk, we only process the first half of data
            in_3_first_chunk = tl.load(in_3_ptr + off_in_3_first, mask=mask_first, other=0.0)
            # Only store the valid portion for first chunk
            first_chunk_mask = (tl.arange(0, BLOCK_SIZE_N) < n3_half) & (off_in_3_first + tl.arange(0, BLOCK_SIZE_N) < n0 * n1 * n2_in3 * n3_half)
            if first_chunk_mask.any():
                tl.store(out_9_ptr + off_out_9, in_3_first_chunk, mask=first_chunk_mask)
    
    # Second chunk (second half of last dimension) 
    if n >= n3_half:
        # Adjust n for second half
        n_second = n - n3_half
        off_in_3_second = m * BLOCK_SIZE_M * n2_in3 * n3 + n_second * BLOCK_SIZE_N + n3_half * n0 * n1 * n2_in3
        off_out_10 = off_in_3_second
        
        mask_second = off_in_3_second < n0 * n1 * n2_in3 * n3
        if mask_second:
            # For the second chunk, we offset by n3_half
            second_chunk_mask = (tl.arange(0, BLOCK_SIZE_N) < n3_half) & (off_in_3_second + tl.arange(0, BLOCK_SIZE_N) < n0 * n1 * n2_in3 * n3)
            if second_chunk_mask.any():
                in_3_second_chunk = tl.load(in_3_ptr + off_in_3_second, mask=second_chunk_mask, other=0.0)
                tl.store(out_10_ptr + off_out_10, in_3_second_chunk, mask=second_chunk_mask)

@torch.fx.wrap
def fused_slicing_chunk_ops(in_0, in_1, in_3, X):
    # Get input shapes
    n0_1, n1_1, n2_1, n3_1 = in_1.shape  # Shape after slicing
    n0_3, n1_3, n2_3, n3_3 = in_3.shape   # Shape of in_3
    
    # Create output tensors
    out_6 = torch.empty((n0_1, n1_1, n2_1, n3_1), dtype=in_1.dtype, device=in_0.device)  # tmp_6
    
    # tmp_7 has same shape as tmp_6 (in_3 * sliced in_0)
    out_7 = torch.empty((n0_1, n1_1, n2_1, n3_1), dtype=in_3.dtype, device=in_0.device)
    
    # Chunks of in_3 have shape [n0_3, n1_3, n2_3, n3_3//2]
    n3_half = n3_3 // 2
    out_9 = torch.empty((n0_3, n1_3, n2_3, n3_half), dtype=in_3.dtype, device=in_0.device)  # tmp_9
    out_10 = torch.empty((n0_3, n1_3, n2_3, n3_half), dtype=in_3.dtype, device=in_0.device)  # tmp_10
    
    # Set up grid dimensions
    BLOCK_SIZE_M = 16  # Block size for first dimensions
    BLOCK_SIZE_N = 1024  # Block size for last dimension
    
    num_m_slice = (n0_1 * n1_1 * n2_1 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_slice = (n3_1 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    num_m_full = (n0_3 * n1_3 * n2_3 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_chunk = (n3_half + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with appropriate grid size
    fused_slicing_chunk_kernel[(num_m_slice, max(num_n_slice, num_n_chunk))](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_3_ptr=in_3,
        out_6_ptr=out_6,
        out_7_ptr=out_7,
        out_9_ptr=out_9,
        out_10_ptr=out_10,
        n0=n0_1, n1=n1_1, n2_slice=n2_1, n3=n3_1,
        n2_in3=n2_3, n3_half=n3_half,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out_6, out_7, out_9, out_10

def replacement_func():
    return fused_slicing_chunk_ops