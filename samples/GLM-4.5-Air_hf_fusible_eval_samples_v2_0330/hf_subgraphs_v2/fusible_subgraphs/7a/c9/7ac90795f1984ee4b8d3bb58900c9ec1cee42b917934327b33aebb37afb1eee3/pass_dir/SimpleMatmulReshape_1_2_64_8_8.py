import torch
import triton
import triton.language as tl

@triton.jit
def simple_matmul_reshape_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_batch,
    n_channels,
    n_inner,
    final_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple fused matmul + reshape kernel"""
    pid = tl.program_id(0)
    
    # Total number of programs = number of elements in output tensor
    total_elements = n_batch * n_channels * final_cols
    
    if pid >= total_elements:
        return
    
    # Calculate output position
    output_row = pid // final_cols
    output_col = pid % final_cols
    
    # Calculate batch and channel indices
    batch_idx = output_row // n_channels
    channel_idx = output_row % n_channels
    
    # Calculate memory offsets for matmul operands
    # a_ptr: [n_batch, n_inner, 1] -> offset = batch_idx * n_inner + channel_idx * n_inner
    # b_ptr: [n_batch, n_channels, n_inner] -> offset = batch_idx * n_channels * n_inner + channel_idx * n_inner
    a_offset = batch_idx * n_inner + channel_idx * n_inner
    b_offset = batch_idx * n_channels * n_inner + channel_idx * n_inner
    
    # Load data from a and b tensors - we know the exact dimensions
    a_indices = tl.arange(0, n_inner)
    b_indices = tl.arange(0, n_inner)[:, None]
    
    a_data = tl.load(a_ptr + a_offset + a_indices)
    b_data = tl.load(b_ptr + b_offset + b_indices)
    
    # Perform matrix multiplication: [n_inner] @ [n_inner, 1] -> scalar
    result = tl.sum(a_data * b_data)
    
    # Store result
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def simple_matmul_reshape(a, b):
    """Simple fused matmul + reshape operation"""
    n_batch, n_channels, n_inner = a.shape
    final_cols = 16  # Target reshape dimension
    
    # Create output tensor
    out_shape = (n_batch * n_channels, final_cols)
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    total_elements = n_batch * n_channels * final_cols
    n_programs = total_elements
    
    simple_matmul_reshape_kernel[(n_programs,)](
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
    """Match the matmul + reshape pattern"""
    matmul_result = torch.matmul(b, a)
    reshape_result = torch.reshape(matmul_result, [-1, 16])
    return reshape_result

def replacement_args(a, b):
    return (a, b)

def replacement_func():
    def kernel_wrapper(a, b):
        return simple_matmul_reshape(a, b)
    return kernel_wrapper