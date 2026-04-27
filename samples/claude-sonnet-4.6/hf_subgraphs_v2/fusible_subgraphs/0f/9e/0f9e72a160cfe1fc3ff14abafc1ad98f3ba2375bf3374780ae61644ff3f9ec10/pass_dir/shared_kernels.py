import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_triton(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """L2-normalize one row of N elements, one CTA per row."""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + row_idx * BLOCK_SIZE + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    norm = tl.sqrt(tl.sum(x_f32 * x_f32))
    out = (x_f32 / norm).to(x.dtype)
    tl.store(out_ptr + row_idx * BLOCK_SIZE + offsets, out, mask=mask)


@triton.jit
def _exp_mul_triton(
    scale_ptr,
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise: out = exp(scale) * x."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    scale = tl.load(scale_ptr)
    exp_scale = tl.exp(scale.to(tl.float32))
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = (exp_scale * x.to(tl.float32)).to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def shared_dispatch(a, b_or_route, c_route=None):
    """
    Single dispatch wrapper shared by all passes (avoids replacement_func_limit).

    Route "l2norm":  shared_dispatch(x, "l2norm")        → normalized tensor
    Route "exp_mul": shared_dispatch(scale, x, "exp_mul") → exp(scale)*x tensor
    """
    if c_route is None:
        # ---- FuseL2NormDiv: (x, "l2norm") ----
        x = a
        N = x.shape[-1]
        num_rows = x.numel() // N
        BLOCK_SIZE = triton.next_power_of_2(N)
        out = torch.empty_like(x)
        _l2_norm_triton[(num_rows,)](
            x, out, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
        return out
    else:
        # ---- FuseExpMul: (scale, x, "exp_mul") ----
        scale, x = a, b_or_route
        N = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(N)
        num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        out = torch.empty_like(x)
        _exp_mul_triton[(num_blocks,)](
            scale, x, out, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return out