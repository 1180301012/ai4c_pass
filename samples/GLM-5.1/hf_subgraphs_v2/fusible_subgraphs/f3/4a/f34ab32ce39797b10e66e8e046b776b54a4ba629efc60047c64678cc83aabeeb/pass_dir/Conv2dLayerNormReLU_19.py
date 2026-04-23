import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(conv2d, (19, 1, 1), in_3, in_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_19")


@triton.jit
def fused_conv_ln_relu_kernel(
    input_ptr, weight_conv_ptr, bias_conv_ptr,
    weight_ln_ptr, bias_ln_ptr, output_ptr,
    N, C_in, C_out, eps,
    stride_in_n, stride_in_ci,
    stride_wc_co, stride_wc_ci,
    stride_bc,
    stride_wln, stride_bln,
    stride_out_n, stride_out_co,
    BLOCK_CIN: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    n = tl.program_id(0)

    cout_off = tl.arange(0, BLOCK_COUT)
    cout_mask = cout_off < C_out

    # Initialize with conv bias
    conv_out = tl.load(bias_conv_ptr + cout_off * stride_bc, mask=cout_mask, other=0.0).to(tl.float32)

    # Accumulate conv2d: loop over C_in blocks
    for ci_start in range(0, C_in, BLOCK_CIN):
        ci_off = ci_start + tl.arange(0, BLOCK_CIN)
        ci_mask = ci_off < C_in

        in_vals = tl.load(input_ptr + n * stride_in_n + ci_off * stride_in_ci,
                          mask=ci_mask, other=0.0).to(tl.float32)

        wc_off = cout_off[:, None] * stride_wc_co + ci_off[None, :] * stride_wc_ci
        wc_mask = cout_mask[:, None] & ci_mask[None, :]
        wc_vals = tl.load(weight_conv_ptr + wc_off, mask=wc_mask, other=0.0).to(tl.float32)

        conv_out += tl.sum(wc_vals * in_vals[None, :], axis=1)

    # Layer norm over C_out channels for batch n
    masked_conv = tl.where(cout_mask, conv_out, 0.0)
    mean = tl.sum(masked_conv, axis=0) / C_out
    diff = tl.where(cout_mask, conv_out - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C_out
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize + affine + ReLU
    normalized = (conv_out - mean) * rstd
    ln_w = tl.load(weight_ln_ptr + cout_off * stride_wln, mask=cout_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(bias_ln_ptr + cout_off * stride_bln, mask=cout_mask, other=0.0).to(tl.float32)
    result = ln_w * normalized + ln_b
    result = tl.maximum(result, 0.0)

    # Store output
    tl.store(output_ptr + n * stride_out_n + cout_off * stride_out_co, result, mask=cout_mask)


@torch.fx.wrap
def fused_conv_ln_relu_dispatch(bias_conv, weight_conv, bias_ln, weight_ln, input, route):
    N = input.shape[0]
    C_in = input.shape[1]

    if route == "route_16":
        C_out = 16
    elif route == "route_19":
        C_out = 19
    elif route == "route_38":
        C_out = 38
    elif route == "route_128":
        C_out = 128
    else:
        C_out = weight_conv.shape[0]

    output = torch.empty((N, C_out, 1, 1), dtype=input.dtype, device=input.device)

    BLOCK_CIN = 64
    BLOCK_COUT = 128

    grid = (N,)

    fused_conv_ln_relu_kernel[grid](
        input, weight_conv, bias_conv, weight_ln, bias_ln, output,
        N=N, C_in=C_in, C_out=C_out, eps=1e-05,
        stride_in_n=input.stride()[0], stride_in_ci=input.stride()[1],
        stride_wc_co=weight_conv.stride()[0], stride_wc_ci=weight_conv.stride()[1],
        stride_bc=bias_conv.stride()[0],
        stride_wln=weight_ln.stride()[0], stride_bln=bias_ln.stride()[0],
        stride_out_n=output.stride()[0], stride_out_co=output.stride()[1],
        BLOCK_CIN=BLOCK_CIN, BLOCK_COUT=BLOCK_COUT,
    )

    return output


def replacement_func():
    return fused_conv_ln_relu_dispatch