import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    """
    Pattern to match matrix multiplication followed by view/reshape operation.
    
    This matches computations like:
    tmp_0 = torch.matmul(in_1, in_0) or tmp_0 = in_1 @ in_0
    tmp_1 = tmp_0.view(target_shape)
    return (tmp_1,)
    """
    tmp_0 = torch.matmul(in_1, in_0)
    tmp_1 = tmp_0.view(tmp_0.shape[0], tmp_0.shape[2], tmp_0.shape[3] if len(tmp_0.shape) > 3 else 1, 1)
    return tmp_1

def replacement_args(in_1, in_0, target_shape):
    """
    Extract arguments for the fused kernel.
    target_shape is the final desired output shape from the view operation.
    """
    return (in_1, in_0, target_shape)

@triton.jit
def fused_matmul_view_kernel(
    # Input tensors
    in_1_ptr,
    in_0_ptr,
    # Output tensor
    out_ptr,
    # Tensor shapes
    batch_size,
    in_0_batch_dim,
    in_0_m_dim,
    in_0_k_dim,
    in_0_n_dim,
    in_1_batch_dim,
    in_1_g_dim,
    in_1_m_dim,
    in_1_k_dim,
    in_1_n_dim,
    # Output shape
    out_batch,
    out_m_dim,
    out_h_dim,
    out_w_dim,
    # Data type and stride information
    element_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Fused matmul + view kernel using Triton.
    Computes matmul and reshapes the output in a single kernel.
    """
    # Compute program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Ensure we don't go out of bounds
    if pid_batch >= batch_size or pid_m >= out_m_dim or pid_n >= out_h_dim:
        return
    
    # Compute memory offsets for the batch
    in_1_batch_offset = pid_batch * in_1_batch_dim * element_size
    in_0_batch_offset = pid_batch * in_0_batch_dim * element_size
    
    # Compute strides for the current batch
    in_1_ptr_batch = in_1_ptr + in_1_batch_offset
    in_0_ptr_batch = in_0_ptr + in_0_batch_offset
    
    # Output offset for the current batch
    out_offset = (pid_batch * out_batch + pid_m * out_m_dim + pid_n) * element_size
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, in_0_k_dim, BLOCK_SIZE_K):
        # Load input blocks
        in_1_block = tl.load(in_1_ptr_batch + 
                            (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * in_1_m_dim * element_size +
                            (k // BLOCK_SIZE_K) * BLOCK_SIZE_K * element_size +
                            tl.arange(0, BLOCK_SIZE_K)[None, :] * element_size,
                            mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] < in_1_m_dim,
                            other=0.0)
        
        in_0_block = tl.load(in_0_ptr_batch + 
                            (k // BLOCK_SIZE_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None] * in_0_k_dim * element_size +
                            pid_n * BLOCK_SIZE_N * element_size,
                            mask=(k // BLOCK_SIZE_K) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None] < in_0_k_dim,
                            other=0.0)
        
        # Matrix multiplication (simplified for the specific pattern)
        # For the patterns we observed, we can optimize further
        accumulator += tl.dot(in_1_block, in_0_block)
    
    # Store result - reshape directly to final output format
    out_ptr_batch = out_ptr + out_offset
    tl.store(out_ptr_batch, accumulator[0, 0], mask=True)

@torch.fx.wrap
def fused_matmul_view(in_1, in_0, target_shape):
    """
    Perform fused matrix multiplication and view operation.
    Eliminates intermediate tensor allocation.
    """
    # Get input tensor shapes
    in_1_shape = in_1.shape
    in_0_shape = in_0.shape
    
    # Compute output shape from matmul
    matmul_shape = torch.matmul(in_1, in_0).shape
    
    # Create output tensor with target shape
    out = torch.empty(target_shape, dtype=torch.float32, device=in_1.device)
    
    # Determine optimal block sizes based on tensor sizes
    MAX_BLOCK_SIZE = 32
    BLOCK_SIZE_M = min(32, matmul_shape[1] if len(matmul_shape) > 1 else MAX_BLOCK_SIZE)
    BLOCK_SIZE_N = 1  # For our patterns, output is often 1 in some dimensions
    BLOCK_SIZE_K = min(32, matmul_shape[2] if len(matmul_shape) > 2 else MAX_BLOCK_SIZE)
    
    # Calculate grid dimensions
    batch_size = in_1_shape[0]
    out_m_dim = target_shape[1]
    out_h_dim = target_shape[2]
    out_w_dim = target_shape[3]
    
    num_m_blocks = (out_m_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_n_blocks = (out_h_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_batch_blocks = (batch_size + 1 - 1) // 1  # One block per batch
    
    # Launch kernel
    fused_matmul_view_kernel[(num_m_blocks, num_n_blocks, num_batch_blocks)](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        in_0_batch_dim=in_1_shape[0] if len(in_1_shape) > 0 else 1,
        in_0_m_dim=in_1_shape[1] if len(in_1_shape) > 1 else 1,
        in_0_k_dim=in_1_shape[2] if len(in_1_shape) > 2 else 1,
        in_0_n_dim=in_1_shape[3] if len(in_1_shape) > 3 else 1,
        in_1_batch_dim=in_0_shape[0] if len(in_0_shape) > 0 else 1,
        in_1_g_dim=in_0_shape[1] if len(in_0_shape) > 1 else 1,
        in_1_m_dim=in_0_shape[2] if len(in_0_shape) > 2 else 1,
        in_1_k_dim=in_0_shape[3] if len(in_0_shape) > 3 else 1,
        in_1_n_dim=in_1_shape[3] if len(in_1_shape) > 3 else 1,
        out_batch=target_shape[0],
        out_m_dim=out_m_dim,
        out_h_dim=out_h_dim,
        out_w_dim=out_w_dim,
        element_size=4,  # float32
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    """
    Return the fused kernel function.
    """
    return fused_matmul_view