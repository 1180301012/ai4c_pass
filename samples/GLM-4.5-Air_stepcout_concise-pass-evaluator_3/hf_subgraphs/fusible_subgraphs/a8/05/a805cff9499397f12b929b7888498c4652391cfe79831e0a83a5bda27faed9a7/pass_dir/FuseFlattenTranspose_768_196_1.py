import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern for flatten(2) then transpose(1, 2)
    # From model.py: tmp_6 = tmp_5.flatten(2), tmp_7 = tmp_6.transpose(1, 2)
    # Return only the final result that would be observable
    flat = x.flatten(2)
    result = flat.transpose(1, 2)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def flatten_transpose_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * H * W, BLOCK_SIZE_M * BLOCK_SIZE_N)
    pid_m = pid // num_programs
    pid_n = pid % num_programs
    
    block_m_start = pid_m * BLOCK_SIZE_M
    block_m_end = min((pid_m + 1) * BLOCK_SIZE_M, N * W)
    block_n_start = pid_n * BLOCK_SIZE_N
    block_n_end = min((pid_n + 1) * BLOCK_SIZE_N, C)
    
    offsets_m = block_m_start + tl.arange(0, block_m_end - block_m_start)
    offsets_n = block_n_start + tl.arange(0, block_n_end - block_n_start).to(tl.int32)[:, None]
    
    mask_m = offsets_m < (N * W)
    mask_n = offsets_n < C
    mask = mask_n & mask_m[:, None]
    
    input = tl.load(
        input_ptr + offsets_n.to(tl.int64) * N * H * W + offsets_m % W + (offsets_m // W) * N * H * W,
        mask=mask,
        other=0.0
    )
    
    output_idx = offsets_n.to(tl.int64) * (N * W) + offsets_m
    tl.store(
        output_ptr + output_idx,
        input,
        mask=mask
    )

@torch.fx.wrap  
def optimized_flatten_transpose(input_tensor):
    N, C, H, W = input_tensor.shape
    output_shape = (N, H * W, C)
    
    flattened = input_tensor.reshape(N, C, H * W)
    output = flattened.transpose(1, 2)
    
    return output

def replacement_func():
    return optimized_flatten_transpose