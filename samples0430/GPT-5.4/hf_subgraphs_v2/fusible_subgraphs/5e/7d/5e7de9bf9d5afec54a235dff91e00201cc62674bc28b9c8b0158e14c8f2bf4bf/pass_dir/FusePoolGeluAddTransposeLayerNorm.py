import torch
import triton
import triton.language as tl


# Match the pre-layernorm subgraph exactly and return the transposed tensor.
def pattern(conv_out, in_3):
    tmp_4 = torch.nn.functional.gelu(conv_out)
    tmp_5 = torch.avg_pool1d(in_3, (2,), (2,), (0,), False, True)
    tmp_6 = tmp_5[(Ellipsis, slice(None, 124, None))]
    tmp_7 = tmp_4[(Ellipsis, slice(None, 124, None))]
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    return tmp_9


def replacement_args(conv_out, in_3):
    return (conv_out, in_3, conv_out, in_3, "fuse_pre_ln")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_C": 256}, num_warps=8),
    ],
    key=["C"],
)
@triton.jit
def _fused_pre_ln_kernel(
    conv_ptr,
    inp_ptr,
    out_ptr,
    conv_stride_b,
    conv_stride_c,
    conv_stride_l,
    inp_stride_b,
    inp_stride_c,
    inp_stride_l,
    out_stride_b,
    out_stride_t,
    out_stride_c,
    C,
    T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = pid_row // T
    t = pid_row % T

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_c < C

    conv = tl.load(
        conv_ptr + b * conv_stride_b + offs_c * conv_stride_c + t * conv_stride_l,
        mask=mask,
        other=0,
    ).to(tl.float32)
    inp0 = tl.load(
        inp_ptr + b * inp_stride_b + offs_c * inp_stride_c + (2 * t) * inp_stride_l,
        mask=mask,
        other=0,
    ).to(tl.float32)
    inp1 = tl.load(
        inp_ptr + b * inp_stride_b + offs_c * inp_stride_c + (2 * t + 1) * inp_stride_l,
        mask=mask,
        other=0,
    ).to(tl.float32)

    gelu = 0.5 * conv * (1.0 + tl.erf(conv * 0.7071067811865475))
    y = 0.5 * (inp0 + inp1) + gelu

    tl.store(
        out_ptr + b * out_stride_b + t * out_stride_t + offs_c * out_stride_c,
        y,
        mask=mask,
    )


@torch.fx.wrap
def shared_replacement_dispatch(x0, x1, x2, x3, route):
    if route == "fuse_pre_ln":
        conv_out = x0
        in_3 = x1
        bsz = conv_out.shape[0]
        channels = conv_out.shape[1]
        out = torch.empty((bsz, 124, channels), device=conv_out.device, dtype=conv_out.dtype)
        grid = lambda META: (bsz * 124, triton.cdiv(channels, META["BLOCK_C"]))
        _fused_pre_ln_kernel[grid](
            conv_out,
            in_3,
            out,
            conv_out.stride(0),
            conv_out.stride(1),
            conv_out.stride(2),
            in_3.stride(0),
            in_3.stride(1),
            in_3.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            channels,
            T=124,
        )
        return out
    if route == "identity_dropout_rand":
        return x0
    return x0


def replacement_func():
    return shared_replacement_dispatch