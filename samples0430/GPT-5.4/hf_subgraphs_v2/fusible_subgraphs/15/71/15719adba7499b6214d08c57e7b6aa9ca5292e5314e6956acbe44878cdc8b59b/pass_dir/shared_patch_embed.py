import torch
import triton
import triton.language as tl

ROUTE_PATCH2_C16_WS2 = "patch2_c16_ws2"
ROUTE_PATCH4_C96_WS8 = "patch4_c96_ws8"


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=["N_TOKENS"],
)
@triton.jit
def _ln_tokens_from_nchw_kernel(
    x_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    out_ptr,
    N_TOKENS,
    SPATIAL,
    C: tl.constexpr,
):
    pid = tl.program_id(0)
    token_id = pid
    if token_id >= N_TOKENS:
        return

    c = tl.arange(0, C)
    x_offsets = c * SPATIAL + token_id
    x = tl.load(x_ptr + x_offsets).to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) / C
    inv_std = tl.rsqrt(var + 1e-5)

    w = tl.load(ln_weight_ptr + c).to(tl.float32)
    b = tl.load(ln_bias_ptr + c).to(tl.float32)
    y = centered * inv_std
    y = y * w + b

    out_offsets = token_id * C + c
    tl.store(out_ptr + out_offsets, y)


def _run_patch2_c16_ws2(conv_out, ln_bias, ln_weight):
    if not conv_out.is_contiguous():
        conv_out = conv_out.contiguous()
    if not ln_weight.is_contiguous():
        ln_weight = ln_weight.contiguous()
    if not ln_bias.is_contiguous():
        ln_bias = ln_bias.contiguous()

    n_tokens = 256
    out = torch.empty((1, n_tokens, 16), device=conv_out.device, dtype=conv_out.dtype)
    _ln_tokens_from_nchw_kernel[(n_tokens,)](
        conv_out,
        ln_weight,
        ln_bias,
        out,
        n_tokens,
        256,
        C=16,
    )
    windows = out.view(1, 16, 16, 16).view(1, 8, 2, 8, 2, 16).permute(0, 1, 3, 2, 4, 5)
    return out, windows


def _run_patch4_c96_ws8(conv_out, ln_bias, ln_weight):
    if not conv_out.is_contiguous():
        conv_out = conv_out.contiguous()
    if not ln_weight.is_contiguous():
        ln_weight = ln_weight.contiguous()
    if not ln_bias.is_contiguous():
        ln_bias = ln_bias.contiguous()

    n_tokens = 65536
    out = torch.empty((1, n_tokens, 96), device=conv_out.device, dtype=conv_out.dtype)
    _ln_tokens_from_nchw_kernel[(n_tokens,)](
        conv_out,
        ln_weight,
        ln_bias,
        out,
        n_tokens,
        65536,
        C=96,
    )
    windows = out.view(1, 256, 256, 96).view(1, 32, 8, 32, 8, 96).permute(0, 1, 3, 2, 4, 5)
    return out, windows


@torch.fx.wrap
def patch_embed_ln_partition_dispatch(conv_out, ln_bias, ln_weight, route):
    if route == ROUTE_PATCH2_C16_WS2:
        return _run_patch2_c16_ws2(conv_out, ln_bias, ln_weight)
    if route == ROUTE_PATCH4_C96_WS8:
        return _run_patch4_c96_ws8(conv_out, ln_bias, ln_weight)
    raise ValueError(f"Unsupported route: {route}")