import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        # ── large M (batch≥32) ───────────────────────────────────────────
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        # BLOCK_K=128 – fewer K-loop iterations for K=512
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        # ── small/medium M (batch=1..8) ──────────────────────────────────
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_avgpool2x2_kernel(
    input_ptr,   # [N_batch, C_in, H_in, W_in]  contiguous NCHW
    weight_ptr,  # [C_out, C_in, 1, 1]  contiguous
    output_ptr,  # [N_batch, C_out, H_out, W_out] contiguous NCHW
    M,           # N_batch * H_out * W_out
    N,           # C_out
    K,           # C_in
    H_out, W_out,
    C_in, W_in,
    HW_in,       # H_in * W_in
    CHW_in,      # C_in * H_in * W_in
    HW_out,      # H_out * W_out
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start   = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_start   = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # Decode linear spatial index → (batch, oh, ow)
    n_idx  = m_offsets // HW_out
    rem    = m_offsets % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out
    ih0    = oh_idx * 2
    iw0    = ow_idx * 2

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        mask   = m_mask[:, None] & k_mask[None, :]

        # ── Load 4 input positions (native dtype, fused avg pool) ─────────
        base   = n_idx[:, None] * CHW_in + k_offsets[None, :] * HW_in
        row0   = base + ih0[:, None] * W_in + iw0[:, None]
        row1   = base + (ih0[:, None] + 1) * W_in + iw0[:, None]

        a00 = tl.load(input_ptr + row0,     mask=mask, other=0.0)
        a01 = tl.load(input_ptr + row0 + 1, mask=mask, other=0.0)
        a10 = tl.load(input_ptr + row1,     mask=mask, other=0.0)
        a11 = tl.load(input_ptr + row1 + 1, mask=mask, other=0.0)

        # Pooled input tile [BLOCK_M, BLOCK_K] — keep in native dtype
        a = (a00 + a01 + a10 + a11) * 0.25

        # ── Load weight tile [BLOCK_K, BLOCK_N] (native dtype) ───────────
        # b[k_local, n_local] = weight[n_offsets[n_local], k_offsets[k_local]]
        w_ptrs = k_offsets[:, None] + n_offsets[None, :] * C_in
        w_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(weight_ptr + w_ptrs, mask=w_mask, other=0.0)

        # fp16/bf16 → tensor cores via out_dtype; fp32 → TF32 tensor cores
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # ── Store ─────────────────────────────────────────────────────────────
    out_ptrs = (n_idx[:, None] * (N * HW_out) +
                n_offsets[None, :] * HW_out +
                oh_idx[:, None] * W_out +
                ow_idx[:, None])

    tl.store(output_ptr + out_ptrs,
             acc.to(output_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


@torch.fx.wrap
def fused_conv1x1_avgpool2x2(weight, x):
    """
    weight : [C_out, C_in, 1, 1]
    x      : [N, C_in, H_in, W_in]   H_in and W_in must be even
    returns: [N, C_out, H_in//2, W_in//2]
    """
    N_batch, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    H_out = H_in // 2
    W_out = W_in // 2

    output = torch.empty((N_batch, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    M     = N_batch * H_out * W_out
    K     = C_in
    HW_in  = H_in * W_in
    CHW_in = C_in * HW_in
    HW_out = H_out * W_out

    # weight is [C_out, C_in, 1, 1] — same memory layout as [C_out, C_in] (last dims are 1)
    # pass directly; kernel indexes as weight_ptr[oc * C_in + ic]
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(C_out, META['BLOCK_N']),
    )

    fused_conv1x1_avgpool2x2_kernel[grid](
        x, weight, output,
        M, C_out, K,
        H_out, W_out,
        C_in, W_in,
        HW_in, CHW_in, HW_out,
    )

    return output


def replacement_func():
    return fused_conv1x1_avgpool2x2