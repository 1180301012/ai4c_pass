import torch
import triton
import triton.language as tl

@triton.jit
def fused_matmul_reshape_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_batch,
    n_channels,
    n_inner,
    final_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused matmul + reshape kernel"""
    pid = tl.program_id(0)
    
    # Each program handles a portion of the batch * channels
    batch_channel_idx = pid
    
    # Calculate original matrix indices
    batch_idx = batch_channel_idx // n_channels
    channel_idx = batch_channel_idx % n_channels
    
    # Load matrix parts
    a_offset = batch_idx * n_channels * n_inner + channel_idx * n_inner
    b_offset = batch_idx * n_inner * 1
    
    # Load a slice [1, n_channels, n_inner] -> [n_inner]
    a_slice = tl.load(a_ptr + a_offset + tl.arange(0, n_inner))
    
    # Load b slice [1, n_inner, 1] -> [n_inner, 1]
    b_slice = tl.load(b_ptr + b_offset + tl.arange(0, n_inner)[:, None])
    
    # Matrix multiplication: [n_inner] @ [n_inner, 1] -> [1]
    result = tl.sum(a_slice[:, None] * b_slice, axis=0)
    
    # Store result in reshaped format
    output_offset = batch_channel_idx * final_cols
    tl.store(out_ptr + output_offset, result[0])

@torch.fx.wrap
def fused_matmul_reshape(a, b, final_cols):
    """Fused matrix multiplication with reshape operation"""
    # Get tensor shapes
    n_batch, n_channels, n_inner = a.shape
    
    # Output size
    total_elements = n_batch * n_channels
    out_size = total_elements * final_cols
    
    # Create output tensor
    out = torch.empty((total_elements, final_cols), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    n_programs = total_elements
    
    # Block sizes optimized for small matrices
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8 
    BLOCK_SIZE_K = 8
    
    fused_matmul_reshape_kernel[(n_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_batch=n_batch,
        n_channels=n_channels,
        n_inner=n_inner,
        final_cols=final_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

def pattern(a, b):
    matmul = torch.matmul(b, a)  # Note: order matters based on computation graph
    tmp_1 = torch.reshape(matmul, [-1, 16])  # 16 is the target reshape dimension
    return tmp_1

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    def kernel_wrapper(a, b):
        return fused_matmul_reshape(a, b, 16)
    return kernel_wrapper