import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3

# Argument extraction function

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_2, in_3, in_1, in_0)

# Triton kernel implementation
@triton.jit
def layer_norm_kernel(in_2_ptr, in_3_ptr, weight_ptr, bias_ptr,
                     out_ptr, N, D, eps, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    start_idx = block_id * D
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    in_2_data = tl.load(in_2_ptr + (start_idx + offsets), mask=mask, other=0.0)
    in_3_data = tl.load(in_3_ptr + (start_idx + offsets), mask=mask, other=0.0)
    x = in_2_data + in_3_data

    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    out = (x - mean) * inv_std * weight + bias
    tl.store(out_ptr + (start_idx + offsets), out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_layer_norm(in_2, in_3, weight, bias, eps=1e-05):
    D = in_2.shape[-1]
    N = in_2.numel() // D
    out = torch.empty_like(in_2)
    BLOCK_SIZE = 1024
    num_blocks = N

    layer_norm_kernel[(num_blocks,)](
        in_2, in_3, weight, bias,
        out, N, D, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# Replacement function

def replacement_func():
    return optimized_layer_norm