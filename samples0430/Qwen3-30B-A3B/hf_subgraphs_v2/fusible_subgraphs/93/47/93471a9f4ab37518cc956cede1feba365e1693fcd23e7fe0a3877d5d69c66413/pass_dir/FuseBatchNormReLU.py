import torch
import triton
import triton.language as tl


def pattern(x, mean, var, weight, bias, training, momentum, eps):
    bn = torch.nn.functional.batch_norm(x, mean, var, weight, bias, training, momentum, eps)
    relu = torch.nn.functional.relu(bn, inplace=False)
    return bn, relu

def replacement_args(x, mean, var, weight, bias, training, momentum, eps):
    return (x, mean, var, weight, bias, training, momentum, eps)

@triton.jit
def batchnorm_relu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    H,
    W,
    BLOCK_C: tl.constexpr
):
    h = tl.program_id(0)
    w = tl.program_id(1)
    c = tl.thread_id(0)

    if c >= C or h >= H or w >= W:
        return

    idx = c * (H * W) + h * W + w
    x_val = tl.load(x_ptr + idx)
    mean_val = tl.load(mean_ptr + c)
    var_val = tl.load(var_ptr + c)
    weight_val = tl.load(weight_ptr + c)
    bias_val = tl.load(bias_ptr + c)

    x_norm = (x_val - mean_val) / tl.sqrt(var_val + 1e-05)
    y = weight_val * x_norm + bias_val
    relu_y = tl.where(y > 0, y, 0.0)

    tl.store(out_ptr + idx, relu_y)


@torch.fx.wrap
def batchnorm_relu_wrapper(x, mean, var, weight, bias, training, momentum, eps):
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    batchnorm_relu_kernel[(H, W), (512,)](
        x, mean, var, weight, bias, out,
        C, H, W,
        BLOCK_C=512
    )
    return out

def replacement_func():
    return batchnorm_relu_wrapper