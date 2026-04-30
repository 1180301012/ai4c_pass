import torch
import triton
import triton.language as tl


def pattern(input, running_mean, running_var, weight, bias):
    result = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    return result


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)


@triton.jit
def bn_inference_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (N*C, num_spatial_blocks)
    nc_idx = tl.program_id(0)
    spatial_block = tl.program_id(1)
    c = nc_idx % C

    # Load BN parameters in float32 for precision
    mean_val = tl.load(mean_ptr + c).to(tl.float32)
    var_val = tl.load(var_ptr + c).to(tl.float32)
    w_val = tl.load(weight_ptr + c).to(tl.float32)
    b_val = tl.load(bias_ptr + c).to(tl.float32)

    # Precompute scale and shift
    inv_std = tl.rsqrt(var_val + 0.001)
    scale = w_val * inv_std
    shift = b_val - mean_val * scale

    # Process spatial elements
    base = nc_idx * HW
    offsets = spatial_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    x = tl.load(input_ptr + base + offsets, mask=mask).to(tl.float32)
    out = x * scale + shift
    tl.store(output_ptr + base + offsets, out, mask=mask)


@torch.fx.wrap
def bn_inference(input, running_mean, running_var, weight, bias):
    device = input.device
    # Only transfer if not already on CUDA
    if not running_mean.is_cuda:
        running_mean = torch.as_tensor(running_mean, device=device)
        running_var = torch.as_tensor(running_var, device=device)
        weight = torch.as_tensor(weight, device=device)
        bias = torch.as_tensor(bias, device=device)

    N, C, H, W = input.shape
    HW = H * W
    NC = N * C

    output = torch.empty_like(input)

    # Select BLOCK_SIZE based on problem characteristics
    if NC <= 256:
        # Small NC: need more spatial parallelism
        BLOCK = 256
    elif HW <= 512:
        # Small spatial: one block per (n,c) is fine
        BLOCK = 1024
    else:
        # Large spatial + large NC: use 1024 for good balance
        BLOCK = 1024

    num_spatial = (HW + BLOCK - 1) // BLOCK
    bn_inference_kernel[(NC, num_spatial)](
        input, output,
        running_mean, running_var, weight, bias,
        C, HW,
        BLOCK_SIZE=BLOCK,
    )
    return output


def replacement_func():
    return bn_inference