import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, tmp_4):
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, tmp_4):
    return (in_0, in_1, in_2, in_3, tmp_4)


@triton.jit
def fused_bn_silu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel's spatial elements
    c_idx = tl.program_id(0)

    # Load BN parameters in float32 for numerical stability
    mean = tl.load(mean_ptr + c_idx).to(tl.float32)
    var = tl.load(var_ptr + c_idx).to(tl.float32)
    weight = tl.load(weight_ptr + c_idx).to(tl.float32)
    bias = tl.load(bias_ptr + c_idx).to(tl.float32)

    # Precompute scale and shift: output = input * scale + shift
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    scale = weight * inv_std
    shift = bias - mean * scale

    # Process spatial elements for this channel
    base_offset = c_idx * spatial_size
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size

    # Load input and convert to float32
    x = tl.load(x_ptr + base_offset + offsets, mask=mask).to(tl.float32)

    # Fused BN + SiLU: silu(bn(x)) = bn(x) * sigmoid(bn(x))
    normalized = x * scale + shift
    sigmoid_val = tl.sigmoid(normalized)
    result = normalized * sigmoid_val

    # Store result (auto-converts back to output dtype)
    tl.store(out_ptr + base_offset + offsets, result, mask=mask)


@torch.fx.wrap
def fused_bn_silu(mean, var, bias, weight, x):
    # x has shape [1, C, H, W] (after reshape in the graph)
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    spatial_size = H * W

    # Ensure parameters are on the correct device
    device = x.device
    mean = mean.to(device)
    var = var.to(device)
    weight = weight.to(device)
    bias = bias.to(device)

    # Make sure input is contiguous for flat indexing
    x = x.contiguous()

    # Allocate output
    out = torch.empty_like(x)

    # Compute BLOCK_SIZE (next power of 2 >= spatial_size)
    BLOCK_SIZE = triton.next_power_of_2(spatial_size)

    # Launch kernel: one program per channel
    grid = (C,)
    fused_bn_silu_kernel[grid](
        x, mean, var, weight, bias, out,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_bn_silu