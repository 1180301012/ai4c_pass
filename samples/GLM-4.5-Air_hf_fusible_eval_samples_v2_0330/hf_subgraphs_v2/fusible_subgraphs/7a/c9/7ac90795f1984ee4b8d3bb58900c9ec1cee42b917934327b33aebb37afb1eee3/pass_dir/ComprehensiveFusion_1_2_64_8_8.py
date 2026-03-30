import torch
import triton
import triton.language as tl

@triton.jit
def comprehensive_fusion_kernel(
    a_ptr,           # in_0: [N, 9, 1]
    b_ptr,           # in_1: [N, C, 9] 
    c_ptr,           # in_2: [1, A, B, C]
    matmul_reshape_out_ptr,  # [N*C, 16]
    transpose_out_ptr,       # [1, A, D, B] where D is original last dim
    n_batch,
    n_channels,
    n_inner,
    final_cols,
    c_dims0, c_dims1, c_dims2, c_dims3,
    BLOCK_SIZE: tl.constexpr,
):
    """Comprehensive fusion of matmul+reshape and transpose operations"""
    pid = tl.program_id(0)
    
    # Total programs = matmul_reshape_elements + transpose_elements
    matmul_reshape_elements = n_batch * n_channels * final_cols
    transpose_elements = c_dims0 * c_dims1 * c_dims2 * c_dims3
    total_elements = matmul_reshape_elements + transpose_elements
    
    if pid >= total_elements:
        return
    
    if pid < matmul_reshape_elements:
        # Handle matmul+reshape part
        # Calculate output position
        output_row = pid // final_cols
        output_col = pid % final_cols
        
        # Calculate original batch and channel indices
        batch_idx = output_row // n_channels
        channel_idx = output_row % n_channels
        
        # Matrix indices
        a_offset = batch_idx * n_channels * n_inner + channel_idx * n_inner
        b_offset = batch_idx * n_inner * 1
        
        # Load data and compute matmul
        a_slice = tl.load(a_ptr + a_offset + tl.arange(0, n_inner))
        b_slice = tl.load(b_ptr + b_offset + tl.arange(0, n_inner)[:, None])
        result = tl.sum(a_slice[:, None] * b_slice)
        
        # Store in matmul+reshape output
        tl.store(matmul_reshape_out_ptr + pid, result)
        
    else:
        # Handle transpose part (pid starts from matmul_reshape_elements)
        transpose_pid = pid - matmul_reshape_elements
        
        # Calculate indices in output (transposed) space
        pid_dim3 = transpose_pid // (c_dims0 * c_dims1 * c_dims2)
        pid_remaining = transpose_pid % (c_dims0 * c_dims1 * c_dims2)
        
        pid_dim2 = pid_remaining // (c_dims0 * c_dims1)
        pid_remaining = pid_remaining % (c_dims0 * c_dims1)
        
        pid_dim1 = pid_remaining // c_dims0
        pid_dim0 = pid_remaining % c_dims0
        
        # Calculate original input indices (swap last two dimensions)
        input_offset = (pid_dim0 * c_dims1 * c_dims2 * c_dims3 + 
                       pid_dim2 * c_dims2 * c_dims3 + 
                       pid_dim3 * c_dims3 + 
                       pid_dim1)
        
        # Store in transpose output
        data = tl.load(c_ptr + input_offset)
        tl.store(transpose_out_ptr + transpose_pid, data)

@torch.fx.wrap
def comprehensive_fusion(in_0, in_1, in_2):
    """Comprehensive fusion of matmul+reshape and transpose operations"""
    # Get tensor shapes for matmul+reshape part
    n_batch, n_channels, n_inner = in_1.shape  # in_1 is [N, C, 9]
    final_cols = 16  # Target reshape dimension
    
    # Get tensor shapes for transpose part
    c_dims0, c_dims1, c_dims2, c_dims3 = in_2.shape
    
    # Output sizes
    matmul_reshape_elements = n_batch * n_channels * final_cols
    transpose_elements = c_dims0 * c_dims1 * c_dims2 * c_dims3
    total_elements = matmul_reshape_elements + transpose_elements
    
    # Create output tensors
    matmul_reshape_out = torch.empty((n_batch * n_channels, final_cols), 
                                   dtype=in_0.dtype, device=in_0.device)
    
    transpose_out = torch.empty_like(in_2)
    
    # Launch kernel
    n_programs = total_elements
    
    comprehensive_fusion_kernel[(n_programs,)](
        a_ptr=in_0,
        b_ptr=in_1,
        c_ptr=in_2,
        matmul_reshape_out_ptr=matmul_reshape_out,
        transpose_out_ptr=transpose_out,
        n_batch=n_batch,
        n_channels=n_channels,
        n_inner=n_inner,
        final_cols=final_cols,
        c_dims0=c_dims0, c_dims1=c_dims1, c_dims2=c_dims2, c_dims3=c_dims3,
        BLOCK_SIZE=1024,
    )
    
    return (matmul_reshape_out, transpose_out)

def pattern(a, b, c):
    """Match the entire computation graph"""
    matmul = torch.matmul(b, a)
    tmp_1 = torch.reshape(matmul, [-1, 16])
    tmp_2 = c.transpose(-1, -2)
    return (tmp_1, tmp_2)

def replacement_args(a, b, c):
    return (a, b, c)

def replacement_func():
    def kernel_wrapper(in_0, in_1, in_2):
        return comprehensive_fusion(in_0, in_1, in_2)
    return kernel_wrapper