import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Match: scalar multiplication with 0.3535533905932738
    """
    tmp_0 = in_1 * 0.3535533905932738
    return tmp_0


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def scalar_mul_kernel(
    input_ptr,
    output_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized scalar multiplication kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply by scalar
    result = x * scalar
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def transpose_2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim0, dim1, dim2, dim3,
    stride_b_in, stride_d0_in, stride_d1_in, stride_d2_in,
    stride_b_out, stride_d0_out, stride_d1_out, stride_d2_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Optimized transpose kernel for last two dimensions"""
    pid_batch = tl.program_id(0)
    pid_d0 = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # Calculate which batch and dimension 0 we're processing
    batch_idx = pid_batch
    d0_idx = pid_d0
    
    # Calculate block ranges
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Create offset arrays
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    # Masks for boundary conditions
    m_mask = m_offsets < dim2
    n_mask = n_offsets < dim3
    
    # Input indices: [batch, d0, m, n]
    input_base = batch_idx * stride_b_in + d0_idx * stride_d0_in
    input_offsets = (
        input_base +
        m_offsets[:, None] * stride_d1_in +
        n_offsets[None, :] * stride_d2_in
    )
    
    # Load with 2D masking
    mask_2d = m_mask[:, None] & n_mask[None, :]
    data = tl.load(input_ptr + input_offsets, mask=mask_2d, other=0.0)
    
    # Output indices: [batch, d0, n, m] (transposed last two dims)
    output_base = batch_idx * stride_b_out + d0_idx * stride_d0_out
    output_offsets = (
        output_base +
        n_offsets[:, None] * stride_d1_out +
        m_offsets[None, :] * stride_d2_out
    )
    
    # Transpose data and store
    data_t = tl.trans(data)
    tl.store(output_ptr + output_offsets, data_t, mask=mask_2d)


@torch.fx.wrap
def optimized_scalar_mul(in_1):
    """
    Optimized scalar multiplication
    """
    scalar_value = 0.3535533905932738
    out = torch.empty_like(in_1)
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    scalar_mul_kernel[grid](
        in_1,
        out,
        scalar_value,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return optimized_scalar_mul