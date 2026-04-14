import torch
import triton
import triton.language as tl


# Single unified kernel:  M=C_out, N=HW_out, K=C_in,  batch = grid dim 2
#   A = W[C_out, C_in]  : K-stride 1  → COALESCED   (no tl.trans → tensor cores ✓)
#   B = X[b, C_in, hw]  : N-stride S  → COALESCED
#   C = out[b, C_out, hw]: N-stride 1  → COALESCED
@triton.autotune(
    configs=[
        # Large HW (≥128): maximize BLOCK_N
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # Small HW (<128): smaller BLOCK_N + larger BLOCK_K for higher arithmetic intensity
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['C_out', 'HW_out', 'C_in'],
    warmup=5,
    rep=10,
)
@triton.jit
def _conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    C_in, H_in, W_in,
    C_out, W_out, HW_out,
    S: tl.constexpr,
    IS_FP16: tl.constexpr, IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid_m = tl.program_id(0)   # C_out tile
    pid_n = tl.program_id(1)   # HW_out tile
    pid_b = tl.program_id(2)   # batch index

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)  # cout range
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)  # hw range

    # Decode hw -> input spatial offset
    h_out    = offs_n // W_out
    w_out_idx = offs_n % W_out
    x_hw   = h_out * (S * W_in) + w_out_idx * S            # [BLOCK_N]
    x_base = pid_b.to(tl.int64) * C_in * H_in * W_in       # scalar
    w_base = offs_m * C_in                                  # [BLOCK_M]

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(C_in, BLOCK_K)):
        offs_k = (k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = offs_k < C_in
        # A = W[cout, cin]:  [BM, BK], K-stride=1 → COALESCED  (NO tl.trans → tensor cores)
        a = tl.load(w_ptr + w_base[:, None] + offs_k[None, :],
                    mask=(offs_m[:, None] < C_out) & mask_k[None, :], other=0.0)
        # B = X[b, cin, hw]: [BK, BN], N-stride=S → COALESCED
        b = tl.load(x_ptr + x_base + offs_k[:, None] * H_in * W_in + x_hw[None, :],
                    mask=mask_k[:, None] & (offs_n[None, :] < HW_out), other=0.0)
        accum = tl.dot(a, b, accum)   # [BM, BK] @ [BK, BN] → [BM, BN]

    offs_cm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    out_ptrs = (out_ptr + pid_b.to(tl.int64) * C_out * HW_out
                + offs_cm[:, None] * HW_out + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < C_out) & (offs_cn[None, :] < HW_out)
    if IS_FP16:
        tl.store(out_ptrs, accum.to(tl.float16), mask=c_mask)
    elif IS_BF16:
        tl.store(out_ptrs, accum.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(out_ptrs, accum, mask=c_mask)


def _launch_conv1x1(x, w, spatial_stride):
    N_b   = x.shape[0]
    C_in  = x.shape[1]
    H_in  = x.shape[2]
    W_in  = x.shape[3]
    C_out = w.shape[0]
    H_out = H_in // spatial_stride
    W_out = W_in // spatial_stride
    HW_out = H_out * W_out

    out = torch.empty((N_b, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    dtype_str = str(x.dtype)
    is_fp16 = (dtype_str == 'torch.float16')
    is_bf16 = (dtype_str == 'torch.bfloat16')

    grid = lambda META: (
        triton.cdiv(C_out,  META['BLOCK_M']),
        triton.cdiv(HW_out, META['BLOCK_N']),
        N_b,
    )
    _conv1x1_kernel[grid](
        x, w, out,
        C_in, H_in, W_in, C_out, W_out, HW_out,
        S=spatial_stride, IS_FP16=is_fp16, IS_BF16=is_bf16,
    )
    return out


@torch.fx.wrap
def conv1x1_route(in_0, in_1, route):
    """Shared replacement: route='s1' (stride 1) or 's2' (stride 2)."""
    if route == 's1':
        return _launch_conv1x1(in_1, in_0, 1)
    else:
        return _launch_conv1x1(in_1, in_0, 2)