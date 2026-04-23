import torch
import triton
import triton.language as tl


# Pattern matching: conv2d -> add 3.0 -> div 6.0 -> clamp(0,1) -> multiply
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2 + 3.0
    tmp_4 = tmp_3 / 6.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "offset3_div6")


# ---- Triton Kernels (identical to offset1_div2 pass, needed for shared routing) ----

@triton.jit
def conv_hardsigmoid_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    N, Cin, Cout,
    offset,
    divisor,
    BLOCK_CIN: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Cout
    oc = pid % Cout

    # Load input row for batch b: input is [N, Cin, 1, 1], offset = b * Cin + ic
    input_offsets = b * Cin + tl.arange(0, BLOCK_CIN)
    input_mask = tl.arange(0, BLOCK_CIN) < Cin
    input_row = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0).to(tl.float32)

    # Load weight row for output channel oc: weight is [Cout, Cin, 1, 1], offset = oc * Cin + ic
    weight_offsets = oc * Cin + tl.arange(0, BLOCK_CIN)
    weight_mask = tl.arange(0, BLOCK_CIN) < Cin
    weight_row = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0).to(tl.float32)

    # Dot product
    dot = tl.sum(input_row * weight_row)

    # Add bias
    bias_val = tl.load(bias_ptr + oc).to(tl.float32)
    dot = dot + bias_val

    # HardSigmoid: clamp((x + offset) / divisor, 0, 1)
    result = (dot + offset) / divisor
    result = tl.maximum(result, 0.0)
    result = tl.minimum(result, 1.0)

    # Store
    tl.store(scale_ptr + pid, result)


@triton.jit
def broadcast_multiply_kernel(
    in2_ptr,
    scale_ptr,
    out_ptr,
    N, Cout, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid_ch = tl.program_id(0)
    pid_sp = tl.program_id(1)

    # Load scale value for this (b, c) - read from float32 buffer
    scale_val = tl.load(scale_ptr + pid_ch).to(tl.float32)

    # Spatial offsets
    spatial_start = pid_sp * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < HW

    # Compute flat offsets in in_2: [N, Cout, H, W] -> b * Cout * HW + c * HW + spatial_offset
    base_offset = pid_ch * HW
    in2_offsets = base_offset + spatial_offsets
    in2_vals = tl.load(in2_ptr + in2_offsets, mask=spatial_mask, other=0.0)

    # Multiply: convert to float32 for computation, result auto-casts to output dtype on store
    out_vals = in2_vals.to(tl.float32) * scale_val

    # Store
    tl.store(out_ptr + in2_offsets, out_vals, mask=spatial_mask)


@torch.fx.wrap
def _fused_se_block(bias, weight, input_tensor, conv_input, offset, divisor):
    # Get shapes (these trigger whitelisted aten.size.default on PosionDispatchTensor)
    N = conv_input.shape[0]
    Cin = conv_input.shape[1]
    Cout = weight.shape[0]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    HW = H * W

    # Allocate scale buffer in float32 for better accuracy
    # torch.empty is whitelisted (aten.empty.memory_format)
    scale = torch.empty((N, Cout), dtype=torch.float32, device=conv_input.device)

    # Allocate output (torch.empty_like is whitelisted)
    output = torch.empty_like(input_tensor)

    # Phase 1: conv + hardsigmoid (no reshape needed - [N,Cin,1,1] has same layout as [N,Cin])
    BLOCK_CIN = triton.next_power_of_2(Cin)
    num_programs_conv = N * Cout
    conv_hardsigmoid_kernel[(num_programs_conv,)](
        input_ptr=conv_input,
        weight_ptr=weight,
        bias_ptr=bias,
        scale_ptr=scale,
        N=N, Cin=Cin, Cout=Cout,
        offset=offset, divisor=divisor,
        BLOCK_CIN=BLOCK_CIN,
    )

    # Phase 2: broadcast multiply
    BLOCK_SIZE = 256
    num_channels = N * Cout
    num_spatial_blocks = triton.cdiv(HW, BLOCK_SIZE)
    broadcast_multiply_kernel[(num_channels, num_spatial_blocks)](
        in2_ptr=input_tensor,
        scale_ptr=scale,
        out_ptr=output,
        N=N, Cout=Cout, HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (output,)


@torch.fx.wrap
def fused_se_block_dispatch(bias, weight, input_tensor, conv_input, route):
    if route == "offset1_div2":
        return _fused_se_block(bias, weight, input_tensor, conv_input, 1.0, 2.0)
    elif route == "offset3_div6":
        return _fused_se_block(bias, weight, input_tensor, conv_input, 3.0, 6.0)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return fused_se_block_dispatch