import torch
import triton
import triton.language as tl


def pattern(input, running_mean, running_var, weight, bias):
    result = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return result


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)


@triton.jit
def bn_eval_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    # Load parameters
    mean = tl.load(mean_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute scale and offset
    scale = w / tl.sqrt(var + 1e-5)
    offset = b - mean * scale

    # Apply to input row
    offs = row * C + cols
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * scale + offset
    tl.store(output_ptr + offs, y, mask=mask)


@torch.fx.wrap
def bn_eval_triton(input, running_mean, running_var, weight, bias):
    output = torch.empty_like(input)
    bn_eval_kernel[(input.shape[0],)](
        input, running_mean, running_var, weight, bias, output,
        input.shape[0], input.shape[1],
        BLOCK_C=512,
        num_warps=1,
        num_stages=1,
    )
    return output


def replacement_func():
    return bn_eval_triton