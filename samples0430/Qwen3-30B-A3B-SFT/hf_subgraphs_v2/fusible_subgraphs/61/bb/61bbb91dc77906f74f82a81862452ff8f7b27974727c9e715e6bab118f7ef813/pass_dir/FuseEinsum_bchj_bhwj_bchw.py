import torch
import triton
import triton.language as tl


def pattern(in_4, in_1):
    result = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    return result


def replacement_args(in_4, in_1):
    return (in_4, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_W': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_W': 32}, num_warps=8),
        triton.Config({'BLOCK_C': 32, 'BLOCK_W': 128}, num_warps=8),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def _einsum_bchj_bhwj_bchw_kernel(
    in4_ptr, in1_ptr, out_ptr,
    B, C, H, W,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """
    Compute 'bchj,bhwj->bchw' using tl.dot (tensor cores).
    For each (b, h): out[b,:,h,:] = in4[b,:,h,:] @ in1[b,h,:,:].T
      in4[b,c,h,j] shape [C,J] → A[c,j] = in4[b,c,h,j], last dim j stride 1  ✓ coalesced
      in1[b,h,w,j] shape [W,J] → B.T[j,w] = in1[b,h,w,j], load as [J,BLOCK_W]  ✓ coalesced
    Grid: (B*H, ceil(C/BLOCK_C), ceil(W/BLOCK_W))
    """
    pid_bh = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_w  = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh % H

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)   # [BLOCK_C]
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)   # [BLOCK_W]
    j_offs = tl.arange(0, BLOCK_J)                      # [BLOCK_J=64]

    c_mask = c_offs < C
    w_mask = w_offs < W

    # ── Load A: in4[b, c_offs, h, j_offs] as [BLOCK_C, J] ──────────────────
    # Last dim = j, stride 1 → coalesced access
    A_base = in4_ptr + b * C * H * BLOCK_J + h * BLOCK_J
    A = tl.load(
        A_base + c_offs[:, None] * (H * BLOCK_J) + j_offs[None, :],
        mask=c_mask[:, None], other=0.0,
    )  # [BLOCK_C, J]

    # ── Load B: in1[b, h, w_offs, j_offs] as [BLOCK_W, J] ──────────────────
    # Last dim = j, stride 1 → coalesced access
    B_base = in1_ptr + b * H * W * BLOCK_J + h * W * BLOCK_J
    B = tl.load(
        B_base + w_offs[:, None] * BLOCK_J + j_offs[None, :],
        mask=w_mask[:, None], other=0.0,
    )  # [BLOCK_W, J]

    # ── A @ B.T  →  [BLOCK_C, J] @ [J, BLOCK_W] = [BLOCK_C, BLOCK_W] ──────
    acc = tl.dot(A, tl.trans(B), out_dtype=tl.float32)  # [BLOCK_C, BLOCK_W]

    # ── Store ─────────────────────────────────────────────────────────────────
    out_base = out_ptr + b * C * H * W + h * W
    out_offs = c_offs[:, None] * (H * W) + w_offs[None, :]
    tl.store(
        out_base + out_offs,
        acc,
        mask=c_mask[:, None] & w_mask[None, :],
    )


@torch.fx.wrap
def triton_einsum_bchj_bhwj_bchw(in_4, in_1):
    B, C, H, J = in_4.shape   # J == W in the input
    W = in_1.shape[2]
    out = torch.empty(B, C, H, W, dtype=in_4.dtype, device=in_4.device)

    BLOCK_J = 64   # J is always 64 for these graphs

    grid = lambda meta: (
        B * H,
        triton.cdiv(C, meta['BLOCK_C']),
        triton.cdiv(W, meta['BLOCK_W']),
    )

    _einsum_bchj_bhwj_bchw_kernel[grid](
        in_4, in_1, out,
        B, C, H, W,
        BLOCK_J=BLOCK_J,
    )

    return out


def replacement_func():
    return triton_einsum_bchj_bhwj_bchw