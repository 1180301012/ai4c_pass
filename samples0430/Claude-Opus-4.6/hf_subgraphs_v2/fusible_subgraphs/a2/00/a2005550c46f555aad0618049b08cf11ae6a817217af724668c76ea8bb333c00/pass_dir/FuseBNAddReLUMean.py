import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def fused_bn_add_relu_mean_kernel(
    input_ptr,
    residual_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_out_ptr,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    c = pid % C

    # Load BN parameters for this channel (compute in float32 for precision)
    rm = tl.load(running_mean_ptr + c).to(tl.float32)
    rv = tl.load(running_var_ptr + c).to(tl.float32)
    w = tl.load(weight_ptr + c).to(tl.float32)
    b = tl.load(bias_ptr + c).to(tl.float32)

    # Precompute scale and shift: output = input * scale + shift
    inv_std = 1.0 / tl.sqrt(rv + 1e-5)
    scale = w * inv_std
    shift = b - rm * scale

    # Base offset for this (n, c) spatial slice
    base_offset = pid * HW

    # Process spatial elements in tiles and accumulate sum for mean
    total_sum = 0.0

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW

        # Load input and residual
        x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + base_offset + offsets, mask=mask, other=0.0).to(tl.float32)

        # Fused BN + Add + ReLU
        bn_out = x * scale + shift
        add_out = r + bn_out
        relu_out = tl.maximum(add_out, 0.0)

        # Zero out invalid positions for correct sum
        relu_valid = tl.where(mask, relu_out, 0.0)

        # Store output
        tl.store(output_ptr + base_offset + offsets, relu_valid, mask=mask)

        # Accumulate sum for mean
        total_sum += tl.sum(relu_valid, axis=0)

    # Compute and store mean
    mean_val = total_sum / HW
    tl.store(mean_out_ptr + pid, mean_val)


@torch.fx.wrap
def fused_bn_add_relu_mean(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: running_mean [C]
    # in_1: running_var [C]
    # in_2: bias [C]
    # in_3: weight [C]
    # in_4: input [N, C, H, W]
    # in_5: residual [N, C, H, W]

    N, C, H, W = in_4.shape
    HW = H * W

    output = torch.empty_like(in_4)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_4.dtype, device=in_4.device)

    grid = (N * C,)

    fused_bn_add_relu_mean_kernel[grid](
        in_4, in_5, in_0, in_1, in_3, in_2,
        output, mean_out,
        C, HW,
    )

    return (output, mean_out)


def replacement_func():
    return fused_bn_add_relu_mean