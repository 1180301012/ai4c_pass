import torch
import triton
import triton.language as tl


def pattern(input, running_mean, running_var, weight, bias):
    result = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    return result


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total_elements', 'C'],
)
@triton.jit
def batch_norm_inference_kernel_1d(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    total_elements, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute channel index for each element
    # For shape [N, C, H, W] in row-major, channel = (flat_offset // HW) % C
    channel_idx = (offsets // HW) % C

    # Load batch norm parameters for the corresponding channel
    rmean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0)
    rvar = tl.load(running_var_ptr + channel_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)

    # Compute scale and shift inline
    eps = 0.001
    scale = w / tl.sqrt(rvar + eps)
    shift = b - rmean * scale

    # Load input
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Apply affine transformation: output = scale * input + shift
    output_val = input_val * scale + shift

    # Store output
    tl.store(output_ptr + offsets, output_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['NHW', 'C'],
)
@triton.jit
def batch_norm_inference_kernel_2d(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    C, HW, NHW,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)  # channel index
    pid_spatial = tl.program_id(1)  # spatial block index within channel

    # Load batch norm parameters for this channel (once per program)
    rmean = tl.load(running_mean_ptr + pid_c)
    rvar = tl.load(running_var_ptr + pid_c)
    w = tl.load(weight_ptr + pid_c)
    b = tl.load(bias_ptr + pid_c)

    # Compute scale and shift for this channel (once per program)
    eps = 0.001
    scale = w / tl.sqrt(rvar + eps)
    shift = b - rmean * scale

    # Compute spatial offsets within this channel
    spatial_offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < NHW

    # Map spatial offsets to global offsets
    # For shape [N, C, H, W], element at channel c, batch n, position hw:
    # global_offset = n * C * HW + c * HW + hw
    n = spatial_offsets // HW
    hw = spatial_offsets % HW
    global_offsets = n * C * HW + pid_c * HW + hw

    # Load input
    input_val = tl.load(input_ptr + global_offsets, mask=spatial_mask, other=0.0)

    # Apply affine transformation
    output_val = input_val * scale + shift

    # Store output
    tl.store(output_ptr + global_offsets, output_val, mask=spatial_mask)


@torch.fx.wrap
def fused_batch_norm_inference(input, running_mean, running_var, weight, bias):
    # Ensure all parameter tensors are on the same device as input
    device = input.device
    running_mean = running_mean.to(device=device)
    running_var = running_var.to(device=device)
    weight = weight.to(device=device)
    bias = bias.to(device=device)

    C = running_mean.shape[0]
    N = input.shape[0]
    HW = input.shape[2] * input.shape[3]
    NHW = N * HW
    total_elements = input.numel()

    output = torch.empty_like(input)

    # Use 2D grid for better scale/shift access pattern
    grid = lambda meta: (C, triton.cdiv(NHW, meta['BLOCK_SIZE']))
    batch_norm_inference_kernel_2d[grid](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        C=C,
        HW=HW,
        NHW=NHW,
    )

    return output


def replacement_func():
    return fused_batch_norm_inference