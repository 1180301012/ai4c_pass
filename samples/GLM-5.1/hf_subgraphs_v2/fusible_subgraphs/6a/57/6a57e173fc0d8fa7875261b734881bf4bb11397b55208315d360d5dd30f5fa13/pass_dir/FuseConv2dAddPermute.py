import torch
import triton
import triton.language as tl
import operator


def pattern(in_0, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    return (conv2d,)


def replacement_args(in_0, in_2):
    return (in_0, in_2, "route_conv2d_groups4")


@triton.jit
def conv2d_depthwise_kernel(
    value_ptr,        # in_2: [N, G, S, W]
    weight_ptr,       # in_0: [G, 1, 65, 1]
    output_ptr,       # output: [N, G, S, W]
    N, G, S, W,
    stride_vn, stride_vg, stride_vs, stride_vw,
    stride_wg, stride_ws,
    stride_on, stride_og, stride_os, stride_ow,
    KERNEL_SIZE: tl.constexpr,
    PAD_LEFT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = N * G * S * W
    mask = offsets < total

    # Decompose flat offset into (n, g, s, w) in output layout [N, G, S, W]
    w_idx = offsets % W
    g_idx = (offsets // W) % G
    s_idx = (offsets // (W * G)) % S
    n_idx = offsets // (W * G * S)

    # Compute conv2d: depthwise 1D convolution along S dimension
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in range(KERNEL_SIZE):
        h_in = s_idx + kh - PAD_LEFT
        # Boundary check for padding
        valid = (h_in >= 0) & (h_in < S) & mask

        # Load value[n, g, h_in, w] from in_2
        v_offset = n_idx * stride_vn + g_idx * stride_vg + h_in * stride_vs + w_idx * stride_vw
        v_val = tl.load(value_ptr + v_offset, mask=valid, other=0.0).to(tl.float32)

        # Load weight[g, 0, kh, 0] from in_0
        w_offset = g_idx * stride_wg + kh * stride_ws
        w_val = tl.load(weight_ptr + w_offset).to(tl.float32)

        acc += v_val * w_val

    # Store at output[n, g, s, w]
    o_offset = n_idx * stride_on + g_idx * stride_og + s_idx * stride_os + w_idx * stride_ow
    tl.store(output_ptr + o_offset, acc, mask=mask)


@torch.fx.wrap
def dispatch_wrapper(in_0, in_2, route):
    if route == "route_conv2d_groups4":
        return _conv2d_depthwise(in_0, in_2)
    elif route == "route_conv2d_groups12":
        return _conv2d_depthwise(in_0, in_2)
    else:
        raise ValueError(f"Unknown route: {route}")


def _conv2d_depthwise(in_0, in_2):
    # in_0: weight [G, 1, 65, 1]
    # in_2: value [N, G, S, W]
    # Output: [N, G, S, W] (same shape as input, since padding preserves dimensions)

    N = in_2.shape[0]
    G = in_2.shape[1]
    S = in_2.shape[2]
    W = in_2.shape[3]

    output = torch.empty((N, G, S, W), dtype=in_2.dtype, device=in_2.device)

    total = N * G * S * W
    BLOCK_SIZE = 256
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv2d_depthwise_kernel[(num_programs,)](
        value_ptr=in_2,
        weight_ptr=in_0,
        output_ptr=output,
        N=N, G=G, S=S, W=W,
        stride_vn=in_2.stride()[0], stride_vg=in_2.stride()[1],
        stride_vs=in_2.stride()[2], stride_vw=in_2.stride()[3],
        stride_wg=in_0.stride()[0], stride_ws=in_0.stride()[2],
        stride_on=output.stride()[0], stride_og=output.stride()[1],
        stride_os=output.stride()[2], stride_ow=output.stride()[3],
        KERNEL_SIZE=65,
        PAD_LEFT=32,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return dispatch_wrapper