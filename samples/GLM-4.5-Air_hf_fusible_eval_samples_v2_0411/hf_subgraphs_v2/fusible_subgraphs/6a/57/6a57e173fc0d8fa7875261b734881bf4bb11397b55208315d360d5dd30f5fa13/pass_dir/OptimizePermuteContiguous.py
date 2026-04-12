import torch
import triton
import triton.language as tl

# Pattern matching function for Permute + Contiguous operations
def pattern(tmp_2):
    """
    Matches the Permute + Contiguous pattern:
    - Permute with (0, 2, 1, 3) 
    - Followed by contiguous() operation
    """
    tmp_3 = tmp_2.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    return tmp_4

# Argument extraction function
def replacement_args(tmp_2):
    return (tmp_2,)

# Optimized Triton kernel for fused Permute + Contiguous
@triton.jit
def fused_permute_contiguous_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim1_size,
    dim2_size,
    dim3_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
):
    """
    Fused Permute + Contiguous kernel
    Efficiently performs permute(0, 2, 1, 3) and ensures contiguous layout
    """
    # Get program IDs for 3D work distribution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)
    
    # Calculate ranges for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    p_start = pid_p * BLOCK_SIZE_P
    
    # Create offsets within the block
    offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offsets_p = p_start + tl.arange(0, BLOCK_SIZE_P)
    
    # Calculate input and output offsets for permute(0, 2, 1, 3)
    # Original shape: [batch_size, dim1_size, dim2_size, dim3_size]
    # After permute: [batch_size, dim2_size, dim1_size, dim3_size]
    
    # Input offsets (original order: batch, dim1, dim2, dim3)
    input_offsets = (
        offsets_m[:, None, None] * (dim1_size * dim2_size * dim3_size) +
        offsets_n[None, :, None] * (dim2_size * dim3_size) +
        offsets_p[None, None, :] * dim3_size
    )
    
    # Output offsets (permuted order: batch, dim2, dim1, dim3)
    output_offsets = (
        offsets_m[:, None, None] * (dim2_size * dim1_size * dim3_size) +
        offsets_n[None, :, None] * (dim1_size * dim3_size) +
        offsets_p[None, None, :] * dim3_size
    )
    
    # Load input data with mask
    mask = (
        (offsets_m[:, None, None] < batch_size) &
        (offsets_n[None, :, None] < dim2_size) &
        (offsets_p[None, None, :] < dim3_size)
    )
    
    input_data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Store directly to output with permuted dimension layout
    tl.store(output_ptr + output_offsets, input_data, mask=mask)

@torch.fx.wrap
def fused_permute_contiguous(input_tensor):
    """
    Wrapper function for fused Permute + Contiguous
    """
    # Get input tensor shape
    batch_size, dim1_size, dim2_size, dim3_size = input_tensor.shape
    
    # Set block sizes for Triton
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_P = 32
    
    # Calculate grid dimensions
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (dim2_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_blocks_p = (dim3_size + BLOCK_SIZE_P - 1) // BLOCK_SIZE_P
    
    # Create output tensor with the same shape but potentially different layout
    output_tensor = torch.empty_like(input_tensor, dtype=input_tensor.dtype)
    
    # Launch kernel
    fused_permute_contiguous_kernel[(num_blocks_m, num_blocks_n, num_blocks_p)](
        input_tensor,
        output_tensor,
        batch_size, dim1_size, dim2_size, dim3_size,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_P,
    )
    
    return output_tensor

# Replacement function (returns function reference)
def replacement_func():
    return fused_permute_contiguous