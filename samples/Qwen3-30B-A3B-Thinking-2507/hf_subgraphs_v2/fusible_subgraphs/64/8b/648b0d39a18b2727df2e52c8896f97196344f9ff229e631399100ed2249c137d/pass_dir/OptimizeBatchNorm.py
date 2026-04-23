import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_0, in_1, in_3, in_2):
    out = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return out

def replacement_args(tmp_5, in_0, in_1, in_3, in_2):
    return (tmp_5, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    C,
    H,
    W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < H * W

    mean = tl.load(mean_ptr + c, mask=(c < C))
    var = tl.load(var_ptr + c, mask=(c < C))
    weight = tl.load(weight_ptr + c, mask=(c < C))
    bias = tl.load(bias_ptr + c, mask=(c < C))

    denominator = tl.sqrt(tl.cast(var, tl.float32) + eps)

    input_ptr_channel = input_ptr + n * C * H * W + c * H * W
    input_val = tl.load(input_ptr_channel + offsets, mask=mask, other=0.0)

    normalized = (input_val - mean) / denominator
    scaled = normalized * weight + bias

    tl.store(output_ptr + n * C * H * W + c * H * W + offsets, scaled, mask=mask)

@torch.fx.wrap
def batch_norm_wrapper(input, mean, var, weight, bias):
    N, C, H, W = input.shape
    eps = 1e-05
    BLOCK_SIZE = 256
    num_blocks = (H * W + BLOCK_SIZE - 1) // BLOCK_SIZE

    output = torch.empty_like(input)
    grid = (N * C, num_blocks)

    batch_norm_kernel[grid](
        input, mean, var, weight, bias, output, N, C, H, W, eps, BLOCK_SIZE
    )

    return output

def replacement_func():
    return batch_norm_wrapper