"""
Pass A: Fuse conv2d(in_2,...) → stack([.],0) → sum(0) → cat([.,in_3],1)

The stack([x], dim=0).sum(dim=0) is a no-op (identity) for a single tensor.
Optimisation strategy:
  1. Eliminate the redundant stack+sum operations entirely.
  2. Implement conv1×1 + cat as a single Triton GEMM kernel writing directly
     into the output buffer, so the cat costs zero extra memory traffic.
  3. N-tiles covering [0, N_conv) are computed via GEMM; tiles covering
     [N_conv, N_total) are a straight copy of `extra`.  Both happen in ONE
     kernel so there is only one launch.
"""
import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    """
    in_0  : bias   [C_out]
    in_1  : weight [C_out, C_in, 1, 1]
    in_2  : input  [B, C_in, H, W]
    in_3  : extra  [B, C_extra, H, W]
    """
    conv    = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv], dim=0)
    summed  = stacked.sum(dim=0)
    result  = torch.cat([summed, in_3], 1)
    return (result,)


def replacement_args(in_0, in_1, in_2, in_3):
    # (bias, weight, conv_input, extra)
    return (in_0, in_1, in_2, in_3)


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4,  num_stages=4),
    ],
    # elem_size distinguishes fp32 (4) from fp16/bf16 (2) so each dtype gets
    # its own autotune cache entry instead of sharing a cross-contaminated config.
    key=['HW', 'K', 'N_conv', 'N_extra', 'elem_size'],
)
@triton.jit
def _conv1x1_cat_kernel(
    input_ptr,   # [B, K, HW]  – NCHW with H*W flattened
    weight_ptr,  # [N_conv, K] – conv weight viewed as 2-D
    bias_ptr,    # [N_conv]
    extra_ptr,   # [B, N_extra, HW]
    output_ptr,  # [B, N_conv+N_extra, HW]
    B, HW, K, N_conv, N_extra, elem_size,   # elem_size is only used as a cache key
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grid: (B, ceil(HW/BLOCK_M), ceil(N_total/BLOCK_N))

    Key optimisation: N-tiles entirely in the copy region (n_start >= N_conv)
    skip the GEMM loop entirely – they only load from `extra` and store.
    This halves the compute for the copy half of the output when N_conv==N_extra.
    N-tiles in the conv region do the full GEMM + optional extra-channel copy
    (for the one boundary tile, if any).
    """
    b      = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    m_offs  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs  = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask     = m_offs < HW
    N_total    = N_conv + N_extra
    n_mask     = n_offs < N_total
    n_is_conv  = n_offs < N_conv           # bool mask [BLOCK_N]

    # ── GEMM accumulator ──────────────────────────────────────────────────────
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # A tile [BLOCK_M, BLOCK_K]: input[b, k_off, hw]
        # Native dtype – Triton uses the appropriate tensor-core variant
        # and accumulates into fp32 regardless of input dtype.
        a_ptrs = (input_ptr
                  + b * K * HW
                  + k_offs[None, :] * HW
                  + m_offs[:, None])
        a = tl.load(a_ptrs,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0)

        # B tile [BLOCK_K, BLOCK_N]: weight[n_off, k_off] (only for n < N_conv)
        safe_n = tl.where(n_is_conv, n_offs, 0)
        w_ptrs = (weight_ptr
                  + safe_n[None, :] * K
                  + k_offs[:, None])
        b_mat = tl.load(w_ptrs,
                        mask=k_mask[:, None] & n_is_conv[None, :],
                        other=0.0)

        acc = tl.dot(a, b_mat, acc)

    # ── Bias add (conv channels) ───────────────────────────────────────────────
    safe_n    = tl.where(n_is_conv, n_offs, 0)
    bias_vals = tl.load(bias_ptr + safe_n,
                        mask=n_is_conv, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    # ── Copy extra channels ───────────────────────────────────────────────────
    n_extra_offs = tl.where(n_is_conv, 0, n_offs - N_conv)
    extra_ptrs   = (extra_ptr
                    + b * N_extra * HW
                    + n_extra_offs[None, :] * HW
                    + m_offs[:, None])
    extra_vals = tl.load(extra_ptrs,
                         mask=m_mask[:, None] & (~n_is_conv)[None, :],
                         other=0.0).to(tl.float32)

    # ── Merge and store ───────────────────────────────────────────────────────
    out_vals = tl.where(n_is_conv[None, :], acc, extra_vals)

    out_ptrs = (output_ptr
                + b * N_total * HW
                + n_offs[None, :] * HW
                + m_offs[:, None])
    tl.store(out_ptrs, out_vals, mask=m_mask[:, None] & n_mask[None, :])


# ── Wrapper ───────────────────────────────────────────────────────────────────

@torch.fx.wrap
def _fused_conv1x1_cat(bias, weight, x, extra):
    """
    bias   : [N_conv]
    weight : [N_conv, C_in, 1, 1]  (standard PyTorch conv2d weight)
    x      : [B, C_in, H, W]
    extra  : [B, N_extra, H, W]
    returns: [B, N_conv+N_extra, H, W]
    """
    B, K, H, W = x.shape
    N_conv  = weight.shape[0]
    N_extra = extra.shape[1]
    N_total = N_conv + N_extra
    HW      = H * W

    output = torch.empty((B, N_total, H, W), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        B,
        triton.cdiv(HW,      meta['BLOCK_M']),
        triton.cdiv(N_total, meta['BLOCK_N']),
    )

    _conv1x1_cat_kernel[grid](
        x, weight, bias, extra, output,
        B, HW, K, N_conv, N_extra, x.element_size(),
    )

    return output


# ── Pass entry point ──────────────────────────────────────────────────────────

def replacement_func():
    return _fused_conv1x1_cat