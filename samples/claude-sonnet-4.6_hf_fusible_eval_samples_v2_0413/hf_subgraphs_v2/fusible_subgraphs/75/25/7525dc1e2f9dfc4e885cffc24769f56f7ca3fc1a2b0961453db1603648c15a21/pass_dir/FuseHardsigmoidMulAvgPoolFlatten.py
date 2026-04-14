import torch
import triton
import triton.language as tl


def pattern(x, y):
    """
    Match: hardsigmoid -> mul -> adaptive_avg_pool2d -> flatten -> dropout (no-op)
    x: conv2d output [B, C, 1, 1]
    y: in_2 [B, C, H, W]
    returns: [B, C]
    """
    t = torch.nn.functional.hardsigmoid(x, False)
    m = y * t
    p = torch.nn.functional.adaptive_avg_pool2d(m, 1)
    f = p.flatten(1, -1)
    d = torch.nn.functional.dropout(f, 0.0, False, False)
    return d


def replacement_args(x, y):
    return (x, y)


@triton.autotune(
    configs=[
        # small BLOCK_BC – useful when BC itself is small (B=1 → BC=1024)
        triton.Config({'BLOCK_BC': 16,  'BLOCK_HW': 64},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_BC': 16,  'BLOCK_HW': 128}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_BC': 16,  'BLOCK_HW': 256}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_BC': 32,  'BLOCK_HW': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_BC': 32,  'BLOCK_HW': 128}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_BC': 32,  'BLOCK_HW': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_BC': 32,  'BLOCK_HW': 256}, num_warps=8,  num_stages=3),
        # medium BLOCK_BC
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 256}, num_warps=8,  num_stages=3),
        # large BLOCK_BC – best for B=128 (BC=131072): fewer programs, more work/program
        triton.Config({'BLOCK_BC': 128, 'BLOCK_HW': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_BC': 128, 'BLOCK_HW': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_BC': 128, 'BLOCK_HW': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_BC': 128, 'BLOCK_HW': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_BC': 128, 'BLOCK_HW': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_BC': 128, 'BLOCK_HW': 256}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_BC': 256, 'BLOCK_HW': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_BC': 256, 'BLOCK_HW': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_BC': 256, 'BLOCK_HW': 128}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_BC': 256, 'BLOCK_HW': 128}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_BC': 256, 'BLOCK_HW': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_BC': 512, 'BLOCK_HW': 64},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_BC': 512, 'BLOCK_HW': 64},  num_warps=16, num_stages=3),
        triton.Config({'BLOCK_BC': 512, 'BLOCK_HW': 128}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_BC': 512, 'BLOCK_HW': 256}, num_warps=16, num_stages=2),
    ],
    key=['BC', 'HW'],
)
@triton.jit
def _fused_hardsigmoid_mul_avgpool_kernel(
    x_ptr,    # [BC]    – hardsigmoid input (conv2d output, contiguous)
    y_ptr,    # [BC*HW] – in_2 (contiguous, row = (b*C+c), col = hw)
    out_ptr,  # [BC]    – output
    BC,       # B * C
    HW,       # H * W
    BLOCK_BC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    2-D tiled kernel.
    Each program handles a BLOCK_BC-wide stripe of (b,c) pairs, reducing
    their HW elements in chunks of BLOCK_HW.

    Memory layout (contiguous):
      x[bc]       = x_ptr  + bc
      y[bc, hw]   = y_ptr  + bc * HW + hw
      out[bc]     = out_ptr + bc
    """
    pid = tl.program_id(0)
    bc_start = pid * BLOCK_BC

    bc_offs = bc_start + tl.arange(0, BLOCK_BC)   # [BLOCK_BC]
    bc_mask = bc_offs < BC

    # ── hardsigmoid on x ────────────────────────────────────────────────────
    x_vals = tl.load(x_ptr + bc_offs, mask=bc_mask, other=0.0).to(tl.float32)
    hs = tl.minimum(tl.maximum((x_vals + 3.0) * (1.0 / 6.0), 0.0), 1.0)

    # ── 2-D blocked reduction of y over HW ──────────────────────────────────
    acc = tl.zeros([BLOCK_BC], dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]
        hw_mask = hw_offs < HW

        # 2-D index: [BLOCK_BC, BLOCK_HW]
        y_offs = bc_offs[:, None] * HW + hw_offs[None, :]
        y_mask = bc_mask[:, None] & hw_mask[None, :]

        y_block = tl.load(y_ptr + y_offs, mask=y_mask, other=0.0).to(tl.float32)
        # Reduce the HW axis → [BLOCK_BC]
        acc += tl.sum(y_block, axis=1)

    # ── output = hardsigmoid(x) * mean(y) ───────────────────────────────────
    out_vals = hs * acc / HW
    tl.store(out_ptr + bc_offs, out_vals, mask=bc_mask)


@torch.fx.wrap
def fused_hardsigmoid_mul_avgpool(x, y):
    """
    Replacement for:
        t = hardsigmoid(x, inplace=False)
        m = y * t
        p = adaptive_avg_pool2d(m, 1)
        f = p.flatten(1, -1)
        d = dropout(f, 0.0, False, False)   # no-op
        return d

    x: [B, C, 1, 1]  (conv2d output, contiguous)
    y: [B, C, H, W]  (contiguous)
    returns: [B, C]  (same dtype as y)

    Memory layout:
      x[bc] at x_ptr + bc          (bc = b*C + c)
      y[bc, hw] at y_ptr + bc*HW + hw
      out[b, c] at out_ptr + b*C + c = out_ptr + bc
    """
    B  = y.shape[0]
    C  = y.shape[1]
    H  = y.shape[2]
    W  = y.shape[3]
    HW = H * W
    BC = B * C

    # Allocate output with the correct dtype – uses only torch.empty (whitelisted)
    out = torch.empty(B, C, dtype=y.dtype, device=y.device)

    # Lambda grid: grid size depends on autotune's BLOCK_BC choice
    _fused_hardsigmoid_mul_avgpool_kernel[
        lambda meta: ((BC + meta['BLOCK_BC'] - 1) // meta['BLOCK_BC'],)
    ](
        x, y, out, BC, HW,
    )

    return out


def replacement_func():
    return fused_hardsigmoid_mul_avgpool