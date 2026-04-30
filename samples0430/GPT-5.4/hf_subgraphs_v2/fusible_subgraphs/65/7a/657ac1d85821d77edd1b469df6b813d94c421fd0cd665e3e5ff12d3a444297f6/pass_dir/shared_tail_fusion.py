import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_tail_kernel(
    x_ptr,
    residual_ptr,
    layer_scale_ptr,
    bn_scale_ptr,
    bn_bias_ptr,
    out_ptr,
    n_elements,
    hw,
    c,
    MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    channel_idx = (offsets // hw) % c

    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if MODE == 2:
        out_val = x_val
    else:
        residual_val = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        layer_scale_val = tl.load(layer_scale_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
        out_val = residual_val + x_val * layer_scale_val

    if MODE != 0:
        bn_scale_val = tl.load(bn_scale_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
        bn_bias_val = tl.load(bn_bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
        out_val = out_val * bn_scale_val + bn_bias_val

    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_tail_dispatch(
    x,
    layer_scale_1d,
    bn_scale_1d,
    bn_bias_1d,
    residual,
    route,
):
    out = torch.empty_like(x)

    n_elements = x.numel()
    c = x.shape[1]
    hw = x.shape[2] * x.shape[3]

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if route == "pre_bn":
        fused_tail_kernel[grid](
            x,
            residual,
            layer_scale_1d,
            bn_scale_1d,
            bn_bias_1d,
            out,
            n_elements,
            hw,
            c,
            MODE=0,
        )
        return out
    elif route == "bn_only":
        fused_tail_kernel[grid](
            x,
            residual,
            layer_scale_1d,
            bn_scale_1d,
            bn_bias_1d,
            out,
            n_elements,
            hw,
            c,
            MODE=2,
        )
        return out
    else:
        return out