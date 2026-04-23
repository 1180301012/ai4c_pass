import torch
import triton
import triton.language as tl

@triton.jit
def fused_roll_ln_add_kernel(
    in_ptr,
    bias_ptr,
    weight_ptr,
    add_ptr,
    out_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    SHIFT_H: tl.constexpr,
    SHIFT_W: tl.constexpr,
    N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_idx = tl.program_id(0)
    h = row_idx // W
    w = row_idx % W
    rolled_h = (h + SHIFT_H) % H
    rolled_w = (w + SHIFT_W) % W
    rolled_flat = rolled_h * W + rolled_w

    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    x = tl.load(in_ptr + rolled_flat * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    add_val = tl.load(add_ptr + row_idx * C + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + 1e-05)

    y = x_centered * rstd * weight + bias
    out = add_val + y

    tl.store(out_ptr + row_idx * C + c_offsets, out.to(in_ptr.dtype.element_ty), mask=c_mask)


@torch.fx.wrap
def dispatch_fused_roll_ln_add(bias, weight, add_val, in_3, route):
    in_3_cont = in_3.contiguous()
    dtype = in_3.dtype

    if route == "f16_32_768":
        H, W, C_val = 32, 32, 768
    elif route == "bf16_64_384":
        H, W, C_val = 64, 64, 384
    elif route == "bf16_32_768":
        H, W, C_val = 32, 32, 768
    elif route == "f16_64_384":
        H, W, C_val = 64, 64, 384
    else:
        raise ValueError(f"Unknown route: {route}")

    N = H * W
    BLOCK_C = 1024
    out = torch.empty((1, N, C_val), dtype=dtype, device=in_3.device)

    fused_roll_ln_add_kernel[(N,)](
        in_ptr=in_3_cont,
        bias_ptr=bias,
        weight_ptr=weight,
        add_ptr=add_val,
        out_ptr=out,
        H=H, W=W, C=C_val,
        SHIFT_H=4, SHIFT_W=4,
        N=N,
        BLOCK_C=BLOCK_C,
    )
    return (out,)