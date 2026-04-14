import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Modified pattern to match linear + transpose structure
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    transposed = linear.transpose(1, 2)
    return linear, transposed  # Return both linear and transposed results

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def simple_linear_kernel(
    bias_ptr, weight_ptr, input_ptr, output_ptr, transposed_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offset < S
    n_mask = n_offset < D
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, H, BLOCK_SIZE_M):
        k_block = tl.arange(0, BLOCK_SIZE_M) + k
        k_mask = k_block < H
        
        input_block = tl.load(input_ptr + m_offset[:, None] * H + k_block[None, :], 
                             mask=k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptr + k_block[:, None] * D + n_offset[None, :], 
                              mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        accumulator += tl.dot(input_block, weight_block)
    
    bias = tl.load(bias_ptr + n_offset, mask=n_mask, other=0.0)
    accumulator = accumulator + bias[None, :]
    
    output_block = accumulator.to(tl.float16)
    
    # Store original output
    output_base = pid_m * D + pid_n * S * D
    output_offsets = m_offset[:, None] * D + n_offset[None, :]
    tl.store(output_ptr + output_base + output_offsets, output_block, 
             mask=m_mask[:, None] & n_mask[None, :])
    
    # Store transposed output
    transposed_base = pid_n * S + pid_m * D * S
    transposed_offsets = n_offset[:, None] * S + m_offset[None, :]
    tl.store(transposed_ptr + transposed_base + transposed_offsets, output_block.T, 
             mask=n_mask[:, None] & m_mask[None, :])

@torch.fx.wrap
def simple_linear_with_transpose(in_0, in_1, in_2):
    B, S, H = in_2.shape
    D = in_0.shape[0]
    
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 64
    num_blocks_m = (S + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (D + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create both output tensors
    output = torch.empty((B, S, D), device=in_2.device, dtype=in_2.dtype)
    transposed_output = torch.empty((B, D, S), device=in_2.device, dtype=in_2.dtype)
    
    # Launch kernel to compute both outputs
    simple_linear_kernel[(num_blocks_m, num_blocks_n)](
        bias_ptr=in_0, weight_ptr=in_1, input_ptr=in_2, 
        output_ptr=output, transposed_ptr=transposed_output,
        B=B, S=S, H=H, D=D, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output, transposed_output

def replacement_func():
    return simple_linear_with_transpose