import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Simple pattern for just Linear operation
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return linear

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def simple_linear_kernel(
    bias_ptr,
    weight_ptr, 
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Simple linear kernel: output = input @ weight.T + bias
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Load bias
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Load weight
    weight = tl.load(
        weight_ptr + n_offsets[:, None] * M + m_offsets[None, :],
        mask=n_mask[:, None] & m_mask[None, :],
        other=0.0
    ).to(tl.float32)
    
    # Load input
    input_data = tl.load(
        input_ptr + m_offsets[:, None] * M + n_offsets[None, :],
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0
    ).to(tl.float32)
    
    # Matrix multiplication
    acc = tl.dot(input_data, weight)
    
    # Add bias and store
    output = acc + bias
    tl.store(
        output_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        output,
        mask=m_mask[:, None] & n_mask[None, :]
    )

@torch.fx.wrap
def simple_linear_fusion(in_0, in_1, in_2):
    """
    Wrapper for simple linear fusion
    """
    M, N = in_2.shape[0], in_0.shape[0]
    
    output = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    simple_linear_kernel[grid_m, grid_n](
        in_0, in_1, in_2, output, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return simple_linear_fusion