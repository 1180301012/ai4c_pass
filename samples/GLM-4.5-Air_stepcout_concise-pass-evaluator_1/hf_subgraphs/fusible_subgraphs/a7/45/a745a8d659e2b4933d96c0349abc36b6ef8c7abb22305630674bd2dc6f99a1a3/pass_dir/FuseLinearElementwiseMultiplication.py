import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    """
    Pattern to match linear transformation followed by element-wise multiplication.
    This matches the transformers pattern: torch.nn.functional.linear(y, x, None) followed by z * result
    """
    linear_result = torch.nn.functional.linear(y, x, None)
    result = z * linear_result
    return (result,)


def replacement_args(x, y, z):
    """
    Extract arguments for the replacement function.
    """
    return (x, y, z)


@triton.jit
def fused_linear_elementwise_kernel(
    x_ptr, x_stride_0, x_stride_1,
    y_ptr, y_stride_0, y_stride_1, y_stride_2,
    z_ptr, z_stride_0, z_stride_1, z_stride_2,
    out_ptr, out_stride_0, out_stride_1, out_stride_2,
    batch_size, seq_len, d_in, d_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Triton kernel for fused linear transformation followed by element-wise multiplication.
    Computes: (y @ x.T) * z
    """
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Compute range each program should process
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, d_out)
    
    # Create output tile
    out_tile = tl.zeros((BLOCK_SIZE_M, seq_len), dtype=tl.float32)
    
    # Compute pointers for this program
    x_row_ptr = x_ptr + pid_m * x_stride_0
    
    # Accumulate matrix multiplication result
    for k in range(0, d_in, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, d_in)
        
        # Load y segment [d_out_slice, d_in_slice] -> [d_in_slice, seq_len]
        y_seg = tl.load(
            y_ptr + pid_b * y_stride_0 + pid_s * y_stride_1 + k * y_stride_2,
            (k_end - k, seq_len),
            mask=(k_end - k, seq_len)
        )
        
        # Load x row segment [d_out_slice, d_in_slice]
        x_row = tl.load(
            x_row_ptr + k * x_stride_1,
            (m_end - m_start, k_end - k),
            mask=(m_end - m_start, k_end - k)
        )
        
        # Matrix multiplication: out += x_row @ y_seg.T
        out_tile += tl.dot(x_row, y_seg, trans_b=True)
    
    # Load z for element-wise multiplication [seq_len, d_out] -> [d_out_slice, seq_len]
    z_seg = tl.load(
        z_ptr + pid_b * z_stride_0 + pid_s * z_stride_1 + m_start * z_stride_2,
        (m_end - m_start, seq_len),
        mask=(m_end - m_start, seq_len)
    )
    
    # Apply element-wise multiplication
    out_tile = out_tile * z_seg
    
    # Store result
    tl.store(
        out_ptr + pid_b * out_stride_0 + pid_s * out_stride_1 + m_start * out_stride_2,
        out_tile,
        (m_end - m_start, seq_len)
    )


@torch.fx.wrap
def fused_linear_elementwise_mm(x, y, z):
    """
    Fused linear transformation followed by element-wise multiplication.
    Computes: (y @ x.T) * z
    """
    # Get tensor shapes
    batch_size = y.size(0)
    seq_len = y.size(1)
    d_in = x.size(1)  # x is [d_out, d_in]
    d_out = x.size(0)
    
    # Create output tensor
    output_shape = (batch_size, seq_len, d_out)
    result = torch.empty(output_shape, dtype=y.dtype, device=y.device)
    
    # Set up block sizes for optimal GPU utilization
    BLOCK_SIZE_M = 64   # Output dimension block (d_out)
    BLOCK_SIZE_N = 128  # Sequence length block (seq_len)
    BLOCK_SIZE_K = 32   # Input dimension block (d_in)
    
    # Calculate grid dimensions
    grid_m = (d_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_s = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_b = batch_size
    
    # Launch kernel
    fused_linear_elementwise_kernel[
        (grid_b, grid_s, grid_m)
    ](
        x, x.stride(0), x.stride(1),
        y, y.stride(0), y.stride(1), y.stride(2),
        z, z.stride(0), z.stride(1), z.stride(2),
        result, result.stride(0), result.stride(1), result.stride(2),
        batch_size, seq_len, d_in, d_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return (result,)


def replacement_func():
    """
    Returns the fused function implementation.
    """
    return fused_linear_elementwise_mm