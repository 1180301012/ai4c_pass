import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_gemm_kernel(
    A_ptr,       # input: [N_batch * HW, C_in]
    W_ptr,       # weight: [C_out, C_in] (contiguous)
    bias_ptr,    # bias: [C_out]
    C_ptr,       # output: [N_batch, C_out, HW] with offset into full tensor
    N_batch, M, N, K, C_out, C_total, HW,
    stride_A_row, stride_A_col,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IS_CONV2D_FIRST: tl.constexpr,  # True → conv_input is in_2; False → conv_input is in_3
):
    """
    Computes 1x1 conv as GEMM and writes directly into the concatenated output tensor.
    
    For row pid_m in [0, N_batch * HW / BLOCK_M):
        n      = pid_m * BLOCK_M / HW        (batch index)
        hw_off = (pid_m * BLOCK_M) % HW      (spatial offset within batch)

    For col pid_n in [0, C_out / BLOCK_N):
        c_out = pid_n * BLOCK_N              (output channel start)

    Output address: n * C_total * HW + c_out * HW + hw_off
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Identify which batch and spatial tile
    n      = pid_m * BLOCK_M // HW
    hw_off = pid_m * BLOCK_M - n * HW   # = (pid_m * BLOCK_M) % HW

    m_start = n * HW + hw_off
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M] relative to n*HW
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    # Accumulate in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load A block: input[n, k, hw] → A[m, k] where m = n*HW + hw
        A_offs = m_offs[:, None] * stride_A_row + k_offs[None, :] * stride_A_col
        A_mask = (m_offs[:, None] < (n * HW + HW)) & (k_offs[None, :] < K)
        a = tl.load(A_ptr + A_offs, mask=A_mask, other=0.0)

        # Load W block: weight[c_out, k]  (shape [C_out, C_in])
        W_offs = n_offs[:, None] * K + k_offs[None, :]
        W_mask = (n_offs[:, None] < C_out) & (k_offs[None, :] < K)
        w = tl.load(W_ptr + W_offs, mask=W_mask, other=0.0)

        acc += tl.dot(a, w, out_dtype=tl.float32)

    # Add bias
    bias_vals = tl.load(bias_ptr + n_offs, mask=n_offs < C_out, other=0.0)
    acc += bias_vals[None, :].to(tl.float32)

    # Write to C_ptr which points into the full [N_batch, C_total, HW] output tensor.
    # The conv2d portion occupies columns [conv_ch_offset, conv_ch_offset + C_out).
    if IS_CONV2D_FIRST:
        conv_ch_offset = 0
    else:
        conv_ch_offset = C_out

    c_row_offs = n * C_total * HW + conv_ch_offset * HW + m_offs
    c_col_offs = n_offs[None, :] * HW + hw_off
    C_offs = c_row_offs + c_col_offs
    C_mask = (m_offs[:, None] < n * HW + HW) & (n_offs[None, :] < C_out)
    tl.store(C_ptr + C_offs, acc.to(C_ptr.dtype.element_ty), mask=C_mask)


@triton.jit
def copy_to_cat_kernel(
    src_ptr, dst_ptr,
    N_batch, C_ch, HW,
    src_batch_stride, dst_batch_stride,
    src_batch_offset, dst_batch_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copies a contiguous batch×channel×spatial slice from src into the
    cat-output tensor at the appropriate channel offset.

    src:  [N_batch, C_ch, HW]  contiguous
    dst:  [N_batch, C_total, HW]  contiguous
    src_batch_offset: offset (in elements) into src for each batch
    dst_batch_offset: offset (in elements) into dst for each batch
    """
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = n_offs < N_batch * C_ch * HW

    # Decompose flat index into (n, c, hw)
    hw   = n_offs % HW
    temp = n_offs // HW
    c    = temp % C_ch
    n    = temp // C_ch

    src_offs = n * src_batch_stride + src_batch_offset + c * HW + hw
    dst_offs = n * dst_batch_stride + dst_batch_offset + c * HW + hw

    vals = tl.load(src_ptr + src_offs, mask=mask)
    tl.store(dst_ptr + dst_offs, vals, mask=mask)


@torch.fx.wrap
def fused_conv2d_stack_sum_cat(bias, weight, in2, in3):
    """
    Fused implementation of:
        conv_out = conv2d(conv_input, weight, bias, stride=1, pad=0, dil=1, groups=1)
        tmp3     = stack([conv_out], dim=0)
        tmp4     = tmp3.sum(dim=0)          # identity: stack+sum = no-op
        out      = cat([tmp4, other], 1)    # cat along channel dim

    Arguments:
        bias    : [C_out]
        weight  : [C_out, C_in, 1, 1]
        in2     : [N, C_in, H, W]  (potential conv input)
        in3     : [N, C_other, H, W]  (potential cat input)

    Returns:
        out : [N, C_out + C_other, H, W]
    """
    N_batch = in2.shape[0]
    C_in    = in2.shape[1]
    H       = in2.shape[2]
    W_dim   = in2.shape[3]
    C_out   = weight.shape[0]
    C_other = in3.shape[1]
    C_total = C_out + C_other
    HW      = H * W_dim

    N_total = N_batch * HW
    M, N, K = N_total, C_out, C_in

    # Allocate full cat output [N_batch, C_total, H, W]
    full_out = torch.empty((N_batch, C_total, H, W_dim),
                           device=in2.device, dtype=in2.dtype)

    # ── Step 1: GEMM (1×1 conv) → writes first C_out channels of full_out ──
    is_conv2d_first = (in2.shape[1] == C_in)  # True if in2 is the conv input

    conv_input = in2 if is_conv2d_first else in3
    other      = in3 if is_conv2d_first else in2

    grid_gemm = lambda meta: (triton.cdiv(N_total, meta['BLOCK_M']),
                              triton.cdiv(C_out,   meta['BLOCK_N']))

    conv1x1_gemm_kernel[grid_gemm](
        conv_input, weight, bias, full_out,
        N_batch, M, N, K,
        C_out, C_total, HW,
        conv_input.stride(0), conv_input.stride(1),   # stride_A_row, stride_A_col
        IS_CONV2D_FIRST=is_conv2d_first,
    )

    # ── Step 2: Copy 'other' → remaining C_other channels of full_out ──
    COPY_BLOCK = 1024
    copy_grid  = (triton.cdiv(N_batch * C_other * HW, COPY_BLOCK),)

    if is_conv2d_first:
        src_batch_off = 0
        dst_batch_off = C_out * HW
    else:
        src_batch_off = C_out * HW
        dst_batch_off = 0

    copy_to_cat_kernel[copy_grid](
        other, full_out,
        N_batch, C_other, HW,
        other.stride(0), full_out.stride(0),
        src_batch_off, dst_batch_off,
        BLOCK_SIZE=COPY_BLOCK,
    )

    return full_out


# ── Pattern & replacement API ────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    """
    Matches the full subgraph:
        conv_out = conv2d(in_2, in_1, in_0, (1,1), (0,0), (1,1), 1)
        tmp3     = torch.stack([conv_out], dim=0)
        tmp4     = tmp3.sum(dim=0)
        out      = torch.cat([tmp4, in_3], 1)
        return out
    """
    conv_out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp3     = torch.stack([conv_out], dim=0)
    tmp4     = tmp3.sum(dim=0)
    out      = torch.cat([tmp4, in_3], 1)
    return out


def replacement_args(in_0, in_1, in_2, in_3):
    # Order: bias, weight, in2 (conv input), in3 (cat input)
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_conv2d_stack_sum_cat