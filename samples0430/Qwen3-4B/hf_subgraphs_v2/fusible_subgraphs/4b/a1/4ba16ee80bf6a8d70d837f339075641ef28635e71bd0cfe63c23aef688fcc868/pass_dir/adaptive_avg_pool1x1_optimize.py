import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    return torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def adaptive_avg_pool_kernel(
    input_ptr,
    output_ptr,
    input_shape,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    B, C, H, W = input_shape
    if b >= B or c >= C:
        return

    sum_val = 0.0
    for h in range(5):
        for w in range(5):
            offset = b * (C * 5) + c * 5 + h * 5 + w
            sum_val += tl.load(input_ptr + offset)

    avg_val = sum_val / 25.0
    tl.store(output_ptr + (b * C + c), avg_val)

@torch.fx.wrap
def kernel_wrapper(input_tensor):
    B, C, H, W = input_tensor.shape
    output = torch.empty(B, C, dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_SIZE = 1024
    num_blocks_b = (B + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_c = (C + BLOCK_SIZE - 1) // BLOCK_SIZE
    adaptive_avg_pool_kernel[(num_blocks_b, num_blocks_c)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_shape=(B, C, H, W),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

def replacement_func():
    return kernel_wrapper