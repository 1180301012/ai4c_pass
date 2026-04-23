import torch
import triton
import triton.language as tl


@triton.jit
def ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr, eps, M,
    BS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    off = tl.arange(0, BS)
    mask = off < N

    # Pre-load weight and bias (shared across rows in this program)
    w = tl.load(w_ptr + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + off, mask=mask, other=0.0).to(tl.float32)

    row_start = pid * ROWS_PER_PROGRAM
    for j in range(ROWS_PER_PROGRAM):
        ri = row_start + j
        if ri < M:
            rs = ri * N
            x = tl.load(x_ptr + rs + off, mask=mask, other=0.0).to(tl.float32)
            mean = tl.sum(x, axis=0) / N
            xm = tl.where(mask, x - mean, 0.0)
            var = tl.sum(xm * xm, axis=0) / N
            rstd = 1.0 / tl.sqrt(var + eps)
            xn = xm * rstd
            out = w * xn + b
            tl.store(out_ptr + rs + off, out, mask=mask)


@triton.jit
def rpb_kernel(
    out_ptr,
    TOTAL: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BS
    offsets = block_start + tl.arange(0, BS)
    mask = offsets < TOTAL

    # Decode flat index: idx = p1 * 588 + p2 * 3 + c
    # where 588 = 196 * 3
    p1 = offsets // 588
    rem = offsets % 588
    p2 = rem // 3
    c = rem % 3

    # Compute relative offsets
    dx = (p2 % 14) - (p1 % 14)
    dy = (p2 // 14) - (p1 // 14)

    # Select value based on channel
    dx_f = dx.to(tl.float32)
    dy_f = dy.to(tl.float32)
    val = tl.where(c == 0, dx_f,
                   tl.where(c == 1, dy_f,
                            dx_f * dx_f + dy_f * dy_f))

    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def _triton_ln(bias, weight, x, N, eps):
    M = x.numel() // N
    out = torch.empty_like(x)
    # Compute BLOCK_SIZE: next power of 2 >= N
    BS = 1
    while BS < N:
        BS *= 2
    ROWS_PER_PROGRAM = 4
    grid = ((M + ROWS_PER_PROGRAM - 1) // ROWS_PER_PROGRAM,)
    ln_kernel[grid](
        x_ptr=x, w_ptr=weight, b_ptr=bias, out_ptr=out,
        N=N, eps=eps, M=M, BS=BS, ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
    )
    return out


@torch.fx.wrap
def _triton_rpb(device, dtype):
    # Create output tensor - compute in float32, convert if needed
    out = torch.zeros(1, 196, 196, 3, dtype=torch.float32, device=device)
    TOTAL = 196 * 196 * 3  # = 115248
    BS = 512
    grid = (TOTAL + BS - 1) // BS
    rpb_kernel[(grid,)](
        out_ptr=out,
        TOTAL=TOTAL,
        BS=BS,
    )
    # Convert to target dtype if needed
    if dtype != torch.float32:
        out = out.to(dtype)
    return out


@torch.fx.wrap
def triton_ln(bias, weight, x):
    N = x.shape[-1]
    eps = 1e-06
    return _triton_ln(bias, weight, x, N, eps)


@torch.fx.wrap
def fused_ln_rpb(bias, weight, x):
    """Fused layer_norm + relative position bias computation."""
    N = x.shape[-1]
    eps = 1e-06
    ln_out = _triton_ln(bias, weight, x, N, eps)
    rpb_out = _triton_rpb(x.device, x.dtype)
    return (rpb_out, ln_out)