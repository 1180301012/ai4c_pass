import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: ONLY the einsum('bchj,bhwj->bchw') call.
#
# FX symbolic_trace converts `in_3 += einsum` to `add` (not `iadd`), so we
# cannot match the full sequence in one pass.  We match only the einsum and
# replace it with a Triton batched-GEMM kernel.
# ---------------------------------------------------------------------------

def pattern(in_4, in_1):
    return torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)


def replacement_args(in_4, in_1):
    return (in_4, in_1)


# ---------------------------------------------------------------------------
# Triton GEMM kernel
#
# Computes: out[b,c,h,w] = sum_j  in_4[b,c,h,j] * in_1[b,h,w,j]
# i.e. for each (b,h): out[b,:,h,:] = in_4[b,:,h,:] @ in_1[b,h,:,:].T
#
# Grid: (B*H,  cdiv(C, BLOCK_M),  cdiv(W, BLOCK_N))
#
# in_4 : [B, C, H, J]  strides (C*H*J, H*J, J, 1)   ← C has stride H*J
# in_1 : [B, H, W, J]  strides (H*W*J, W*J, J, 1)
# out  : [B, C, H, W]  strides (C*H*W, H*W, W, 1)
#
# The A matrix (in_4) has non-unit stride in C; we maximise throughput via
# aggressive software pipelining (num_stages) to hide HBM latency.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # BLOCK_K=64: single K-tile (best tensor-core efficiency)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8,  num_stages=4),
        # BLOCK_K=32: two K-tiles → software pipelining hides non-contiguous A latency
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4,  num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8,  num_stages=4),
    ],
    key=['B', 'C', 'H', 'W', 'J'],
)
@triton.jit
def einsum_bchj_bhwj_bchw_kernel(
    in4_ptr, in1_ptr, out_ptr,
    B, C, H, W,
    J: tl.constexpr,        # always 64; constexpr enables loop-unrolling
    # NOTE: strides are NOT passed; they are computed from shapes for contiguous tensors.
    # This reduces kernel arguments and marshaling overhead.
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    # Grid: (B * cdiv(C, BLOCK_M),  H,  cdiv(W, BLOCK_N))
    bc_id = tl.program_id(0)
    h     = tl.program_id(1)
    n_id  = tl.program_id(2)

    num_c_tiles = tl.cdiv(C, BLOCK_M)
    b    = bc_id // num_c_tiles
    m_id = bc_id %  num_c_tiles

    c_offs = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    w_offs = n_id * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Strides for contiguous tensors (computed from shapes)
    # in_4 [B, C, H, J]: strides (C*H*J, H*J, J, 1)
    s4h = J;  s4c = J * H;  s4b = J * H * C
    # in_1 [B, H, W, J]: strides (H*W*J, W*J, J, 1)
    s1w = J;  s1h = J * W;  s1b = J * W * H
    # out  [B, C, H, W]: strides (C*H*W, H*W, W, 1)
    soh = W;  soc = W * H;  sob = W * H * C

    in4_base = in4_ptr + b * s4b + h * s4h
    in1_base = in1_ptr + b * s1b + h * s1h

    for _k in range(0, tl.cdiv(J, BLOCK_K)):
        k_offs = _k * BLOCK_K + tl.arange(0, BLOCK_K)

        # A: [BLOCK_M, BLOCK_K]
        a_ptrs = in4_base + c_offs[:, None] * s4c + k_offs[None, :]
        a_mask = (c_offs[:, None] < C) & (k_offs[None, :] < J)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # BT: [BLOCK_N, BLOCK_K]
        bt_ptrs = in1_base + w_offs[:, None] * s1w + k_offs[None, :]
        bt_mask = (w_offs[:, None] < W) & (k_offs[None, :] < J)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, tl.trans(bt), allow_tf32=True)

    out_mask = (c_offs[:, None] < C) & (w_offs[None, :] < W)
    out_ptrs = out_ptr + b*sob + c_offs[:, None]*soc + h*soh + w_offs[None, :]
    tl.store(out_ptrs, acc.to(OUTPUT_DTYPE), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_einsum_bchj_bhwj_bchw(in_4, in_1):
    """
    Triton replacement for torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1).
    in_4 : [B, C, H, J]
    in_1 : [B, H, W, J]
    out  : [B, C, H, W]
    """
    B, C, H, J = in_4.shape
    _B, _H, W, _J = in_1.shape

    out = torch.empty((B, C, H, W), dtype=in_4.dtype, device=in_4.device)

    _dtype = in_4.dtype
    if _dtype == torch.float16:
        out_tl_dtype = tl.float16
    elif _dtype == torch.bfloat16:
        out_tl_dtype = tl.bfloat16
    else:
        out_tl_dtype = tl.float32

    # Grid: (B * cdiv(C, BLOCK_M),  H,  cdiv(W, BLOCK_N))
    # Grouping consecutive h with same (b, c_tile) → L2 cache reuse of in_4 rows.
    grid = lambda meta: (
        B * triton.cdiv(C, meta['BLOCK_M']),
        H,
        triton.cdiv(W, meta['BLOCK_N']),
    )

    einsum_bchj_bhwj_bchw_kernel[grid](
        in_4, in_1, out,
        B, C, H, W, J,
        OUTPUT_DTYPE=out_tl_dtype,
    )

    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_einsum_bchj_bhwj_bchw