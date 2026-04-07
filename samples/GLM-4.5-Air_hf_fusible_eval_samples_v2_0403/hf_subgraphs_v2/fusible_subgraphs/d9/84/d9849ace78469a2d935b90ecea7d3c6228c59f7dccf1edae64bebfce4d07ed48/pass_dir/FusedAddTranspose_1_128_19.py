import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(a, b):
    """
    Pattern matches: addition followed by transpose(1,2)
    a: [128, 1] tensor that gets broadcasted to [1, 128, 19]
    b: [1, 128, 19] tensor that gets added to broadcasted a
    Returns: (b + a) (broadcasted) then transpose(1,2) to [1, 19, 128]
    """
    # Regular addition with broadcasting (creates new tensor)
    result = b + a  # result has shape [1, 128, 19], a has shape [128, 1] -> broadcasts to [1, 128, 19]
    
    # Assign to match the model's pattern structure
    in_2 = result
    
    # Transpose dims 1 and 2: [1, 128, 19] -> [1, 19, 128]
    tmp_2 = in_2.transpose(1, 2)
    
    return tmp_2


# Argument extraction function  
def replacement_args(a, b):
    return (a, b)


@triton.jit
def fused_add_transpose_kernel(
    a_ptr,          # [128, 1] tensor
    b_ptr,          # [1, 128, 19] tensor  
    out_ptr,
    dim1_size,      # 128 (second dimension of b)
    dim2_size,      # 19 (third dimension of b)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for new dimension 1 (19)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for new dimension 2 (128)
):
    """
    Fused kernel: broadcast addition + transpose
    Input: a [128, 1] -> broadcasts to [1, 128, 19]
           b [1, 128, 19]
    Operation: b + a (broadcasted)
    Output: transpose(b + a, 1, 2) -> [1, 19, 128]
    """
    # Each program handles a 2D tile in the transposed output space
    m = tl.program_id(0)  # dimension 0 (always 0 since dim0=1)
    k = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  
    n = tl.program_id(2) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create bounds check
    k_mask = k < dim2_size
    n_mask = n < dim1_size
    
    # Create coordinate grids
    k_indices = k[:, None]  # [BLOCK_SIZE_K, 1]
    n_indices = n[None, :]  # [1, BLOCK_SIZE_N]
    
    # Load b: [1, 128, 19] -> access original layout at [0, n_indices, k_indices]
    b_offsets = k_indices * dim1_size + n_indices
    b_vals = tl.load(b_ptr + b_offsets, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)
    
    # Load a: [128, 1] -> access at [n_indices, 0] and broadcast to [1, 128, 19]
    a_offset = n_indices  # [1, BLOCK_SIZE_N] flattened offset
    a_vals = tl.load(a_ptr + a_offset, mask=n_mask[None, :], other=0.0)
    
    # Broadcast a_vals along k dimension to match b_vals shape
    a_vals = tl.broadcast_to(a_vals, (BLOCK_SIZE_K, b_vals.shape[1]))
    
    # Perform fused addition: b + a (with broadcasting)
    sum_vals = b_vals + a_vals
    
    # Store result in transposed layout: [1, 19, 128]
    tl.store(out_ptr + b_offsets, sum_vals, mask=(k_mask[:, None] & n_mask[None, :]))


@torch.fx.wrap
def fused_add_transpose(a, b):
    """
    Fused operation: broadcast addition + transpose(1,2)
    Input: a [128, 1], b [1, 128, 19]
    Output: (b + a).transpose(1, 2) -> [1, 19, 128]
    """
    dim1_size = b.shape[1]  # 128
    dim2_size = b.shape[2]  # 19
    
    # Output shape after transpose: [1, 19, 128]
    out_shape = (b.shape[0], dim2_size, dim1_size)
    out = torch.empty(out_shape, dtype=b.dtype, device=b.device)
    
    # Define block sizes (must be power of 2 for tl.arange)
    BLOCK_SIZE_K = 16  # Power of 2, <= 19
    BLOCK_SIZE_N = 32  # Power of 2, <= 128
    
    # Calculate grid dimensions
    m_dim = 1  # always 1 since dim0=1
    k_dim = (dim2_size + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  # 19 // 16 = 2
    n_dim = (dim1_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N  # 128 // 32 = 4
    
    # Launch kernel
    fused_add_transpose_kernel[(m_dim, k_dim, n_dim)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out


# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_add_transpose