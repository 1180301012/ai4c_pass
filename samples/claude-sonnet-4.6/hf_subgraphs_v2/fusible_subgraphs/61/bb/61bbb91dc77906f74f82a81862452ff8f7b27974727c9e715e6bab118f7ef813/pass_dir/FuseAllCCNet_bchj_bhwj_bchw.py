"""
Full-fusion pass for the complete CCNet attention chain:
  einsum = einsum('bchj,bhwj->bchw', in_4, in_1)
  in_3  += einsum          ← operator.iadd
  tmp    = in_3 * in_0     ← scalar scale
  out    = (tmp + in_2).contiguous()

Strategy:
  - Monkey-patch Proxy.__iadd__ to emit proper 'operator.iadd' nodes so the
    pattern can match the model's iadd node.
  - One Triton kernel fuses: batched matmul + iadd + scale + add.
    This avoids materialising 3 intermediate [B,C,H,W] tensors.

Memory savings vs. un-fused original (BF16 B=32, ~134 MB each):
  Original I/O: read(in4,in1) + write(einsum) + read(in3,einsum) +
                write(iadd) + read(iadd,scale) + write(mul) +
                read(mul,in2) + write(add)   ≈ 1,205 MB
  Fused I/O:   read(in4,in1,in3,in2,scale) + write(out)  ≈ 553 MB
"""

import operator
import torch
import torch.fx
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch Proxy.__iadd__ so that `in_3 += einsum` in a pattern function
# creates a call_function(operator.iadd) node instead of falling back to __add__
# ─────────────────────────────────────────────────────────────────────────────
def _proxy_iadd(self, other):
    return self.tracer.create_proxy(
        'call_function',
        operator.iadd,
        (self, other),
        {}
    )

if not getattr(torch.fx.Proxy, '_ai4c_iadd_patched', False):
    torch.fx.Proxy.__iadd__ = _proxy_iadd
    torch.fx.Proxy._ai4c_iadd_patched = True


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: full chain  einsum → iadd → mul → add → contiguous
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_3 += einsum          # now emits operator.iadd (monkey-patched)
    in_5 = in_3
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: batched matmul fused with (in_3 + acc) * scale + in_2
#
# Grid: (B*H,  ceil(C/BLOCK_M),  ceil(W/BLOCK_N))
# Each program:
#   for fixed (b, h):
#     acc[m, n] = sum_k  in4[b, m, h, k]  *  in1[b, h, n, k]
#     out[b, m, h, n] = (in3[b, m, h, n] + acc[m, n]) * scale + in2[b, m, h, n]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64, 'BLOCK_K': 64},
                      num_stages=6, num_warps=2),
    ],
    key=['B', 'C', 'H', 'W', 'J'],
)
@triton.jit
def _full_fused_kernel(
    in4_ptr, in1_ptr, in3_ptr, in0_ptr, in2_ptr, out_ptr,
    B, C, H, W, J,
    in4_s0, in4_s1, in4_s2, in4_s3,   # strides for in_4  [B, C, H, J]
    in1_s0, in1_s1, in1_s2, in1_s3,   # strides for in_1  [B, H, W, J]
    in3_s0, in3_s1, in3_s2, in3_s3,   # strides for in_3  [B, C, H, W]
    in2_s0, in2_s1, in2_s2, in2_s3,   # strides for in_2  [B, C, H, W]
    out_s0, out_s1, out_s2, out_s3,   # strides for out   [B, C, H, W]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh  % H).to(tl.int64)

    m_offs = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    n_offs = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    # Float32 accumulator for the matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    in4_base = b * in4_s0 + h * in4_s2
    in1_base = b * in1_s0 + h * in1_s1

    # Matmul loop over K (single iteration when BLOCK_K = J = 64)
    for k_start in range(0, J, BLOCK_K):
        k_offs = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)

        # A = in_4[b, m, h, k]  →  [BLOCK_M, BLOCK_K]
        a_ptrs = in4_base + m_offs[:, None] * in4_s1 + k_offs[None, :] * in4_s3
        a_mask = (m_offs[:, None] < C) & (k_offs[None, :] < J)
        a = tl.load(in4_ptr + a_ptrs, mask=a_mask, other=0.0)

        # B = in_1[b, h, n, k]  →  [BLOCK_N, BLOCK_K]
        b_ptrs = in1_base + n_offs[:, None] * in1_s2 + k_offs[None, :] * in1_s3
        b_mask = (n_offs[:, None] < W) & (k_offs[None, :] < J)
        b_mat  = tl.load(in1_ptr + b_ptrs, mask=b_mask, other=0.0)

        # acc  +=  A @ B^T   →  [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, tl.trans(b_mat), acc)

    # Load scalar scale
    scale = tl.load(in0_ptr).to(tl.float32)

    # Load in_3 and in_2 element tiles
    in3_base = b * in3_s0 + h * in3_s2
    in2_base = b * in2_s0 + h * in2_s2
    out_base = b * out_s0 + h * out_s2

    m = m_offs[:, None]
    n = n_offs[None, :]
    mask = (m < C) & (n < W)

    in3_val = tl.load(in3_ptr + in3_base + m * in3_s1 + n * in3_s3,
                      mask=mask, other=0.0).to(tl.float32)
    in2_val = tl.load(in2_ptr + in2_base + m * in2_s1 + n * in2_s3,
                      mask=mask, other=0.0).to(tl.float32)

    # Fused epilogue: (in_3 + matmul_result) * scale + in_2
    result = (in3_val + acc) * scale + in2_val

    tl.store(out_ptr + out_base + m * out_s1 + n * out_s3, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-transpose kernel: in4[B,C,H,J] → in4_T[B,H,C,J]
# This converts the C-dimension stride from H*J=4096 to J=64,
# enabling much more cache-friendly access in the main kernel.
# Only beneficial when in4 fits in L2 (small B).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _transpose_in4_kernel(
    src_ptr, dst_ptr,
    B, C, H, J,
    BLOCK_C: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """
    Grid: (B, H, ceil(C / BLOCK_C))
    Each program copies a [BLOCK_C, J] tile from in4 to in4_T.
    src layout: [B, C, H, J]  strides: [C*H*J,  H*J, J, 1]
    dst layout: [B, H, C, J]  strides: [H*C*J,  C*J, J, 1]
    """
    pid_b = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_c = tl.program_id(2).to(tl.int64)

    C_i = tl.cast(C, tl.int64)
    H_i = tl.cast(H, tl.int64)
    J_i = tl.cast(J, tl.int64)

    c_offs = (pid_c * BLOCK_C + tl.arange(0, BLOCK_C)).to(tl.int64)
    j_offs = tl.arange(0, BLOCK_J).to(tl.int64)
    c_mask = c_offs[:, None] < C_i

    # Source: src[b, c, h, j] with strides [C*H*J, H*J, J, 1]
    src_ptrs = (pid_b * (C_i * H_i * J_i)
                + c_offs[:, None] * (H_i * J_i)
                + pid_h * J_i
                + j_offs[None, :])
    data = tl.load(src_ptr + src_ptrs, mask=c_mask, other=0.0)

    # Destination: dst[b, h, c, j] with strides [H*C*J, C*J, J, 1]
    dst_ptrs = (pid_b * (H_i * C_i * J_i)
                + pid_h * (C_i * J_i)
                + c_offs[:, None] * J_i
                + j_offs[None, :])
    tl.store(dst_ptr + dst_ptrs, data, mask=c_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel wrapper
# ─────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def full_fused_ccnet(in_0, in_1, in_2, in_3, in_4):
    """
    Replacement for the full CCNet chain:
      einsum('bchj,bhwj->bchw', in_4, in_1)  →  iadd into in_3
      → scale by in_0  →  add in_2  →  .contiguous()

    For small B (in4 fits in L2), we pre-transpose in4 [B,C,H,J] → [B,H,C,J]
    to make the C-dimension access contiguous (stride J=64 vs H*J=4096).
    """
    B = in_3.shape[0]
    C = in_3.shape[1]
    H = in_3.shape[2]
    W = in_3.shape[3]
    J = in_4.shape[3]

    out = torch.empty_like(in_3)

    # Always use direct in_4 access (transpose overhead exceeds cache benefit)
    in4_ptr = in_4
    in4_s0  = in_4.stride(0)
    in4_s1  = in_4.stride(1)
    in4_s2  = in_4.stride(2)
    in4_s3  = in_4.stride(3)

    grid = lambda meta: (
        B * H,
        triton.cdiv(C, meta['BLOCK_M']),
        triton.cdiv(W, meta['BLOCK_N']),
    )

    _full_fused_kernel[grid](
        in4_ptr, in_1, in_3, in_0, in_2, out,
        B, C, H, W, J,
        in4_s0, in4_s1, in4_s2, in4_s3,
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pass interface
# ─────────────────────────────────────────────────────────────────────────────
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return full_fused_ccnet