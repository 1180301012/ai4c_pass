import torch
from pass_dir.conv1x1_kernels import fused_dispatch


def pattern(in_0, in_1, in_2, in_3):
    """
    Matches: conv2d(in_2, in_1, in_0) → stack([x],0) → sum(0) → cat([x, in_3], 1)
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.stack([conv2d], dim=0)
    tmp_4  = tmp_3.sum(dim=0)
    tmp_5  = torch.cat([tmp_4, in_3], 1)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "route_a")


# Must be at the module level so the import order is clear
def replacement_func():
    return fused_dispatch
#                 out    [N, M, Cout]  → strides [M*Cout, Cout, 1]
#                             reads weight transposed implicitly via tl.trans
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    a_ptr,                      # input    [N, K, M]
    w_ptr,                      # weight   [N, K]   (Cout rows × Cin cols)
    bias_ptr,                   # bias     [N]      (Cout)
    out_ptr,                    # output   [N, M, Cout]
    M, K, N,                   # M=H*W, K=Cin,  N=Cout
    stride_am, stride_ak,       # K-stride = W, Cin-stride = 1
    stride_wn, stride_wk,       # Cin-stride = 1, Cout-stride = Cin (=K)
    stride_om, stride_on,       # stride over M (spatial), stride over Cout
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load input tile [BLOCK_M, BLOCK_K]
        a = tl.load(
            a_ptr
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Load weight tile [BLOCK_K, BLOCK_N]  (transposed load from [N, K])
        w = tl.load(
            w_ptr
            + offs_k[:, None] * stride_wk
            + offs_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # acc += a @ w^T  =>  [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc = tl.dot(a, tl.trans(w), acc)

    # Bias add (accumulate in float32 → cast back to input dtype)
    bias = tl.load(
        bias_ptr + offs_n,
        mask=offs_n < N,
        other=0.0,
    )
    acc += bias[None, :].to(tl.float32)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out_ptr
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on,
        acc.to(a_ptr.dtype.element_ty),
        mask=out_mask,
    )


@torch.fx.wrap
def _fused_conv1x1_stack_sum_cat_b2_c3(in_0, in_1, in_2, in_3):
    """
    Fused: conv2d(in_2, in_1, in_0) → stack([x],0) → sum(0) → cat([x, in_3], 1)

    in_0 : bias   [Cout]
    in_1 : weight [Cout, Cin, 1, 1]
    in_2 : branch [N, Cin, H, W]
    in_3 : other  [N, Cout, H, W]
    returns       [N, Cout, H, W]  (= cat([conv_out, in_3], dim=1))
    """
    N  = in_2.shape[0]
    Cin = in_2.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    Cout = in_0.shape[0]
    M = H * W

    # Output: [N, Cout, H, W]
    out = torch.empty((N, Cout, H, W), dtype=in_2.dtype, device=in_2.device)

    # As 2-D views for GEMM
    # in_2 : [N, Cin, H, W]  → [N, Cin, M]   (spatial flattened → S)
    a      = in_2.contiguous().view(N, Cin, M)
    # in_1 : [Cout, Cin, 1, 1] → [Cout, Cin]
    w      = in_1.view(Cout, Cin)
    # in_0 : [Cout]
    # out   : [N, Cout, H, W] → [N, M, Cout]
    co     = out.view(N, M, Cout)

    # Contiguous copy helpers
    stride_aM = W   #  Cin*S = Cin*H*W; each step in 'M' dimension of a[N,K,M]
    stride_aK = 1   # K-stride = Cin (inner K-loop)
    stride_wN = Cin # weight row-stride
    stride_wK = 1   # weight col-stride
    stride_oM = Cout  # [N,M,Cout] strides
    stride_oN = 1

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Cout, meta['BLOCK_N']),
    )

    _conv1x1_gemm_kernel[grid](
        a, w, in_0, co,
        M, Cin, Cout,
        stride_aM, stride_aK,
        stride_wN, stride_wK,
        stride_oM, stride_oN,
    )

    # ------------------------------------------------------------------
    # Cat: copy in_3 (already in out[:, C1:, :, :])
    # out : [N, Cout, H, W] = [N, C1+C2, H, W]
    # out[:, :C1, :, :] ← conv result (already written above)
    # out[:, C1:, :, :] ← in_3  [N, C2, H, W]
    # ------------------------------------------------------------------
    C1 = Cout                                # channels from conv
    C2 = in_3.shape[1]
    NN = in_3.shape[0]
    H  = in_3.shape[2]
    W2 = in_3.shape[3]
    M2 = H * W2

    out2 = out.view(NN, M2, C1 + C2)   # [N, M, Cout] — same flat layout

    # z = in_3 → out2[:, C1:]
    z   = in_3.contiguous().view(NN, M2, C2)   # [NN, M2, C2]
    dst = out2.view(NN, M2, C1 + C2)

    # B-D view of the "other" part of the output
    col_stride = M2                        # stride between rows in [N,M,C1+C2]
    c_src = tl.arange(0, C2)              # source column index
    c_dst = C1 + c_src                    # destination column index
    full_mask = (tl.arange(0, 1) < NN) & (tl.arange(0, 1) < M2)  # gate condition

    _concat_copy_kernel[(NN * M2,)](
        out2, z, dst,
        col_stride, C2, C1, C1 + C2,
        BLOCK_LEN=1024,
    )

    return (out,)


# ---------------------------------------------------------------------------
# Cat-copy kernel: for each (n, j) pair, copy z[:, j : j+C2] → dst[:, j : j+C2]
# Implemented as a 1-D elementwise kernel over (N*M) groups × C2 elements.
# ---------------------------------------------------------------------------
@triton.jit
def _concat_copy_kernel(
    src_ptr,
    z_ptr,
    dst_ptr,
    col_stride,      # stride between rows in the [N, M, C] view
    C2_src,          # number of elements to copy per group
    C1,              # first "block" width (unused in ptr arithmetic)
    C_total,         # total output channel width = C1 + C2
    BLOCK_LEN: tl.constexpr,
):
    pid = tl.program_id(0)        # pid = n * M + m
    n_m = pid // (C2_src // BLOCK_LEN)
    # ... simplified: just copy C2 elements starting at column n_m*C2_src/??
    # Re-derive: to copy z[:, n_m % M, :] → dst[:, n_m % M, C1:]

    # Simpler: grid = (NN*M2,), each program copies C2_src elements at its group
    col = pid % (C2_src // BLOCK_LEN * BLOCK_LEN)  # over-complicated

    # Efficient 1-D approach: grid = (NN*M2*num_c_blocks, )
    C2_BLOCKS = (C2_src + BLOCK_LEN - 1) // BLOCK_LEN
    # Total elements: NN * M2 * C2_src

    # Sum over all elements with flat index
    n_elements = NN * M2 * C2_src
    BLOCK = BLOCK_LEN
    n_blocks = (n_elements + BLOCK - 1) // BLOCK

    # Use a 1-D grid
    # src[n, m, c]  →  dst[n, m, C1+c]
    # fully_fused_kernel[(n_blocks,)](src_flat, z_flat, dst_flat, n_elements, ...)

    # ---- Practical version: one program per (n, m) row, BLOCK_LEN elements ----
    # pid encodes the row: pid = n * M2 + m
    row = pid
    m = row % M2
    n_idx = row // M2

    c_start = 0
    base_src = n_idx * col_stride * C2_src + m * C2_src + c_start
    base_dst = n_idx * col_stride * C_total + m * C_total + C1

    # Load multiple blocks from src and dst
    for block_start in range(0, C2_src, BLOCK):
        c_slice = c_start + block_start + tl.arange(0, BLOCK)
        mask = c_slice < C2_src
        val = tl.load(z_ptr + base_src + c_slice, mask=mask, other=0.0)
        tl.store(dst_ptr + base_dst + c_slice, val, mask=mask)


# Re-use _concat_copy_kernel via 1-D flat indexing
@triton.jit
def _flat_copy_kernel(
    src_ptr, z_ptr, dst_ptr,
    src_shape_all,   # total elements in src (e.g. N*M*C2)
    col_stride_n,    # stride of one "column" block in the output (= C_total)
    C1,
    C2,
    BLOCK_LEN: tl.constexpr,
):
    """
    Flat 1-D copy: valid for when src and z BOTH represent the [N, M, C2] tensor.
    Each program handles BLOCK_LEN elements of the flat src/z array.
    z element at flat index i goes to dst at flat index i // (C2/col_block) ???
    
    Simpler intuition – work in [N, M, C2] layout:
    z[i] → dst[n * M * C_total + m * C_total + C1 + c]
          = dst[(n * M + m) * C_total + C1 + (i % C2)]
    
    where i = n_with_m * C2 + c  (n_with_m: padded index spanning [0, N*M))
    
    Actually: for i ∈ [0, N*M*C2):
      n = i // (M*C2)
      j = (i // C2) % M     ← this is m
      c = i % C2
    src index: n * M * C2 + j * C2 + c  ≡  i  (continuity assumed)
    dst index: n * M * C_total + j * C_total + C1 + c
    """
    cond = src_shape_all > 0

    pid = tl.program_id(0)
    offs = pid * BLOCK_LEN + tl.arange(0, BLOCK_LEN)
    mask = offs < src_shape_all

    # Decompose flat index into (n, j, c)
    C2 = col_stride_n - C1   # we don't know C2 but it's col_stride_n - C1
    C2_f = C2.to(tl.int64)

    local = offs % C2_f
    rem   = offs // C2_f
    j     = rem % (col_stride_n // C2_f)
    n     = rem // (col_stride_n // C2_f)

    C_total_f = col_stride_n.to(tl.int64)
    dst_idx   = n * col_stride_n * C_total_f + j * C_total_f + C1 + local

    val = tl.load(z_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_idx, val, mask=mask)


@torch.fx.wrap
def fused_conv1x1_stack_sum_cat_b2_c3(in_0, in_1, in_2, in_3):
    N  = in_2.shape[0]
    Cin = in_2.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    Cout = in_0.shape[0]
    M = H * W

    out = torch.empty((N, Cout, H, W), dtype=in_2.dtype, device=in_2.device)

    a    = in_2.contiguous().view(N, Cin, M)
    w    = in_1.view(Cout, Cin)
    co   = out.view(N, M, Cout)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Cout, meta['BLOCK_N']),
    )
    _conv1x1_gemm_kernel[grid](
        a, w, in_0, co,
        M, Cin, Cout,
        W, 1,        # a strides
        Cin, 1,      # w strides (row=Cout stride=Cin, col=Cin stride=1)
        Cout, 1,     # co strides
    )

    # Copy in_3 into the second half of out
    C1 = Cout
    C2 = in_3.shape[1]
    NN = in_3.shape[0]
    H2 = in_3.shape[2]
    W2 = in_3.shape[3]
    M2 = H2 * W2

    # Safe empty ensure we don't hold both views alive
    tmp = out.view(NN, M2, C1 + C2)
    z   = in_3.contiguous().view(NN, M2, C2)
    _flat_copy_kernel[(NN * M2,)](
        z, z, tmp,
        NN * M2 * C2,      # src_shape_all
        C1 + C2,           # col_stride_n
        C1,
        C2,
        BLOCK_LEN=1024,
    )
    return (out,)