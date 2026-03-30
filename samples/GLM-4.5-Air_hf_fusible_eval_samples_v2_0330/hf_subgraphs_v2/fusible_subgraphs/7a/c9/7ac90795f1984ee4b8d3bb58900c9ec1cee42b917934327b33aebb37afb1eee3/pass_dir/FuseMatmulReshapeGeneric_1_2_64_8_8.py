import torch
import triton
import triton.language as tl

@triton.jit
def fused_matmul_reshape_kernel_generic(
    a_ptr,
    b_ptr,
    out_ptr,
    n_batch,
    n_channels,
    n_inner,
    final_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Generic fused matmul + reshape kernel for different reshape dimensions"""
    pid = tl.program_id(0)
    
    # Each program handles one element in the reshaped output
    # The total number of programs equals the total number of elements in the reshaped tensor
    total_elements = n_batch * n_channels * final_cols
    
    if pid >= total_elements:
        return
    
    # Calculate output position
    output_row = pid // final_cols
    output_col = pid % final_cols
    
    # Calculate original batch and channel indices
    batch_idx = output_row // n_channels
    channel_idx = output_row % n_channels
    
    # Matrix indices are fixed for small matrices
    a_offset = batch_idx * n_channels * n_inner + channel_idx * n_inner
    b_offset = batch_idx * n_inner * 1
    
    # Load data
    a_slice = tl.load(a_ptr + a_offset + tl.arange(0, n_inner))
    b_slice = tl.load(b_ptr + b_offset + tl.arange(0, n_inner)[:, None])
    
    # Perform matmul: [n_inner] @ [n_inner, 1] -> scalar
    result = tl.sum(a_slice[:, None] * b_slice)
    
    # Store result
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def fused_matmul_reshape_generic(a, b, final_cols):
    """Generic fused matrix multiplication with reshape operation"""
    # Get tensor shapes
    n_batch, n_channels, n_inner = a.shape
    
    # Output size after reshape
    total_elements = n_batch * n_channels * final_cols
    
    # Create output tensor
    out = torch.empty((n_batch * n_channels, final_cols), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    n_programs = total_elements
    
    # Optimal block size for small matrices
    fused_matmul_reshape_kernel_generic[(n_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_batch=n_batch,
        n_channels=n_channels,
        n_inner=n_inner,
        final_cols=final_cols,
        BLOCK_SIZE=1024,
    )
    
    return out

def pattern(a, b):
    matmul = torch.matmul(b, a)
    reshaped = torch.reshape(matmul, [-1, 16])  # This will be parameterized
    return reshaped

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    def kernel_wrapper(a, b):
        # This handles the case where final_cols = 16
        return fused_matmul_reshape_generic(a, b, 16)
    return kernel_wrapper