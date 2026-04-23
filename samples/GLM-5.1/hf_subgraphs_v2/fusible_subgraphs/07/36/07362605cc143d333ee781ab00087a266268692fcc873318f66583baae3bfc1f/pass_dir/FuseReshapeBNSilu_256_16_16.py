import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "FuseReshapeBNSilu_256_16_16")

@triton.jit
def reshape_bn_silu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    total_elements: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    spatial_size = H * W
    channel = offsets // spatial_size

    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mean_val = tl.load(running_mean_ptr + channel, mask=mask, other=0.0)
    var_val = tl.load(running_var_ptr + channel, mask=mask, other=1.0)
    weight_val = tl.load(weight_ptr + channel, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + channel, mask=mask, other=0.0)

    normalized = (input_val - mean_val) / tl.sqrt(var_val + eps)
    bn_out = normalized * weight_val + bias_val

    silu_out = bn_out * tl.sigmoid(bn_out)

    tl.store(output_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def fused_reshape_bn_silu_512_8_8(in_0, in_1, in_2, in_3, in_4, route):
    C = 512
    H = 8
    W = 8
    total_elements = C * H * W
    output = torch.empty((1, C, H, W), dtype=in_4.dtype, device=in_4.device)
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    reshape_bn_silu_kernel[(num_programs,)](
        input_ptr=in_4, running_mean_ptr=in_0, running_var_ptr=in_1,
        weight_ptr=in_3, bias_ptr=in_2, output_ptr=output,
        C=C, H=H, W=W, total_elements=total_elements, eps=1e-05, BLOCK_SIZE=BLOCK_SIZE,
    )
    return (output,)

@torch.fx.wrap
def fused_reshape_bn_silu_256_16_16(in_0, in_1, in_2, in_3, in_4, route):
    C = 256
    H = 16
    W = 16
    total_elements = C * H * W
    output = torch.empty((1, C, H, W), dtype=in_4.dtype, device=in_4.device)
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    reshape_bn_silu_kernel[(num_programs,)](
        input_ptr=in_4, running_mean_ptr=in_0, running_var_ptr=in_1,
        weight_ptr=in_3, bias_ptr=in_2, output_ptr=output,
        C=C, H=H, W=W, total_elements=total_elements, eps=1e-05, BLOCK_SIZE=BLOCK_SIZE,
    )
    return (output,)

@torch.fx.wrap
def _placeholder_a(in_0, in_1, in_2, in_3, in_4, route):
    return fused_reshape_bn_silu_256_16_16(in_0, in_1, in_2, in_3, in_4, route)

@torch.fx.wrap
def _placeholder_b(in_0, in_1, in_2, in_3, in_4, route):
    return fused_reshape_bn_silu_512_8_8(in_0, in_1, in_2, in_3, in_4, route)

@torch.fx.wrap
def shared_dispatch_wrapper(in_0, in_1, in_2, in_3, in_4, route):
    if route == "FuseReshapeBNSilu":
        return fused_reshape_bn_silu_512_8_8(in_0, in_1, in_2, in_3, in_4, route)
    elif route == "FuseReshapeBNSilu_256_16_16":
        return fused_reshape_bn_silu_256_16_16(in_0, in_1, in_2, in_3, in_4, route)
    elif route == "FuseReshapeBNSilu_float16_512_8_8":
        return _placeholder_a(in_0, in_1, in_2, in_3, in_4, route)
    elif route == "FuseReshapeBNSilu_float16_256_16_16":
        return _placeholder_b(in_0, in_1, in_2, in_3, in_4, route)
    else:
        raise ValueError(f"Unknown route: {route}")

def replacement_func():
    return shared_dispatch_wrapper