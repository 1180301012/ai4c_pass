import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    # First linear: in_5 [300,256] x in_1.T [256,512] + in_0 -> [300,512]
    tmp_4 = torch.nn.functional.linear(in_5, in_1, in_0)
    # Slice first 256 cols, then view to [300,256]
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    # Slice last 256 cols, then view to [300,256]
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    # Second linear: in_4.reshape(300, -1, 256) [300,256] x in_3.T [256,512] + in_2 -> [300,1,512]
    tmp_9 = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    # Slice first 256 last dim -> [300,1,256]
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    # Slice last 256 last dim -> [300,1,256]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    # Unsqueeze tmp_6 -> [1,300,256]
    tmp_13 = tmp_6.unsqueeze(-2)
    return (tmp_11, tmp_12, tmp_8, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# -----------------------------------------------------------------------
# Kernel 1: Compute out1 = in_5 @ in_1.T + in_0  (full [300,512])
#   then write both halves to out1a [300,256] and out1b [300,256]
# in_5 strides: [256, 1]  ->  ptr + m*stride_am + k*stride_ak
# in_1 is weight [512,256] stored as B.T where B[N,K]; B.T access:
#   b[k,n] = ptr + n*K + k  ->  (n_idx*K + k_idx) = n_idx*256 + k_idx
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M1', 'N1', 'K1'],
)
@triton.jit
def fused_linear_split_kernel(
    in5_ptr,   # [M1, K]  float32/float16/bf16
    in1_ptr,   # [N, K]   float32/float16/bf16  (weight matrix)
    in0_ptr,   # [N]      float32/float16/bf16  (bias)
    out1a_ptr, # [M1, N_HALF] first half
    out1b_ptr, # [M1, N_HALF] second half
    M1, N1, K1,
    stride_am, stride_ak,
    stride_in1n, stride_in1k,
    stride_o1a,  stride_o1b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    N_HALF = 256

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first K-tile of A (in_5) and K-tile of B.T (in_1)
    a_ptrs = in5_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = in1_ptr + offs_n[:, None] * stride_in1n + offs_k[None, :] * stride_in1k

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(0, tl.cdiv(K1, BLOCK_K)):
        k_rem = K1 - k_iter * BLOCK_K
        k_mask = offs_k < k_rem

        mask_a = (offs_m[:, None] < M1) & k_mask[None, :]
        mask_b = (offs_n[:, None] < N1) & k_mask[None, :]

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_in1k

    # Add bias
    bias = tl.load(in0_ptr + offs_n, mask=offs_n < N1, other=0.0)
    acc = acc + bias[None, :]

    # Store to both output halves
    half_offsets = tl.arange(0, 256)
    mask_n = offs_n < N1

    store_a = tl.arange(0, 256)
    mask_n1 = offs_n < N_HALF
    store_b = store_a + N_HALF
    mask_n2 = (offs_n >= N_HALF) & mask_n

    out_mask_a = (offs_m[:, None] < M1) & mask_n1[None, :]
    out_mask_b = (offs_m[:, None] < M1) & mask_n2[None, :]

    tl.store(out1a_ptr + offs_m[:, None] * stride_o1a + half_offsets[None, :],
             acc.to(DTYPE), mask=out_mask_a)
    tl.store(out1b_ptr + offs_m[:, None] * stride_o1b + store_b[None, :],
             acc.to(DTYPE), mask=out_mask_b)


# -----------------------------------------------------------------------
# Kernel 2: Compute out2 = in4_flat [300,256] @ in_3.T + in_2   ([300,1,512])
#   in_4 shape [1,150,1,512] stride=[76800,512,512,1]
#   -> in4_flat[m, c] = in4_ptr[c*256 + m]  (m=c*300+r)
#      Proof: in4[0,c,0,k] = in4_ptr[c*512+k], in4_flat[c*300+r, k] = in4_ptr[c*512+r*...]
#      Actually: in4[0,c,0,k] = in4_ptr[c*512+k] = in4_flat[c*300+r, k] when c*512+k = (c*300+r)*? OK check:
#      in4[0, c, 0, k] = in4_ptr[0*76800 + c*512 + 0*512 + k] = in4_ptr[c*512+k]
#      in4_flat[m, k] = in4_flat_ptr[m*256 + k]
#      m = r (after reshape [..., r, k]) where r in [0,149]: in4_flat[r, k] = in4_ptr[r*512+k]
#      This is exactly in4[0, r, 0, k] so in4_flat = in4.view(300, -1, 256).view(300, 256) contiguous
#
#   grid: (ceil(M2/BLOCK_M), ceil(N2/BLOCK_N)) where M2=300, N2=512
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M2', 'N2', 'K2'],
)
@triton.jit
def fused_linear_k2_kernel(
    in4_ptr,    # [300,256] viewed as flat – in4_ptr[m*256 + k]
    in3_ptr,    # [N2, K2]  (weight matrix)
    in2_ptr,    # [N2]      (bias)
    out2a_ptr,  # [300,256] first half -> tmp_11 [300,1,256] already done in view
    out2b_ptr,  # [300,256] second half -> tmp_12 [300,1,256] already done in view
    M2, N2, K2,
    stride_aka,      # in4 row stride = 256
    stride_aka_2,    # out2 row stride = 256
    stride_in3n,     # in3 row stride = K2
    stride_in3k,     # in3 col stride = 1
    stride_o2a,      # out2a row stride = 256
    stride_o2b,      # out2b row stride = 256
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    N_HALF = 256

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # in4_flat: in4_ptr[m*256 + k]  (m ranges over 0..299)
    a_ptrs = in4_ptr + offs_m[:, None] * stride_aka + offs_k[None, :]

    # in3 stored col-major as [N2,K2]: in3[n,k] = ptr + n*K2 + k
    b_ptrs = in3_ptr + offs_n[:, None] * stride_in3n + offs_k[None, :] * stride_in3k

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(0, tl.cdiv(K2, BLOCK_K)):
        k_rem = K2 - k_iter * BLOCK_K
        k_mask = offs_k < k_rem

        mask_a = (offs_m[:, None] < M2) & k_mask[None, :]
        mask_b = (offs_n[:, None] < N2) & k_mask[None, :]

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K          # stride_ak = 1 for in4
        b_ptrs += BLOCK_K * stride_in3n   # stride_in3k = 1, advance K, new N row after K stride

    # Add bias
    bias = tl.load(in2_ptr + offs_n, mask=offs_n < N2, other=0.0)
    acc = acc + bias[None, :]

    # Store to out2a (first halves, shape [300,256]) and out2b (last halves, shape [300,256])
    half_offsets = tl.arange(0, 256)
    mask_n = offs_n < N2
    mask_n1 = offs_n < N_HALF
    mask_n2 = (offs_n >= N_HALF) & mask_n

    out_mask_a = (offs_m[:, None] < M2) & mask_n1[None, :]
    out_mask_b = (offs_m[:, None] < M2) & mask_n2[None, :]

    tl.store(out2a_ptr + offs_m[:, None] * stride_o2a + half_offsets[None, :],
             acc.to(DTYPE), mask=out_mask_a)
    tl.store(out2b_ptr + offs_m[:, None] * stride_o2b + (half_offsets + N_HALF)[None, :],
             acc.to(DTYPE), mask=out_mask_b)


# -----------------------------------------------------------------------
# Shared wrapper
# -----------------------------------------------------------------------
@torch.fx.wrap
def linear_split_fused(in_0, in_1, in_2, in_3, in_4, in_5):
    # Compute dtype for kernel (must match pointer type)
    dtype = in_5.dtype
    if dtype == torch.float16:
        DTYPE = tl.float16
    elif dtype == torch.bfloat16:
        DTYPE = tl.bfloat16
    else:
        DTYPE = tl.float32

    # ---- Compute both solutions in parallel kernel calls ----

    # Kernel 1: out1 = in_5 @ in_1.T + in_0 → two [300,256] outputs
    # in_5: [300, 256], in_1: [512, 256], in_0: [512]
    out1a = torch.empty((300, 256), dtype=dtype, device=in_5.device)
    out1b = torch.empty((300, 256), dtype=dtype, device=in_5.device)

    M1, K1, N1 = 300, 256, 512
    grid1 = lambda META: (
        triton.cdiv(M1, META['BLOCK_M']),
        triton.cdiv(N1, META['BLOCK_N']),
    )

    fused_linear_split_kernel[grid1](
        in_5, in_1, in_0,
        out1a, out1b,
        M1, N1, K1,
        in_5.stride(0), in_5.stride(1),
        in_1.stride(0), in_1.stride(1),
        out1a.stride(0), out1b.stride(0),
        DTYPE=DTYPE,
    )

    # Kernel 2: out2 = in4_flat [300,256] @ in_3.T + in_2
    # in_4: [1,150,1,512] → viewed as flat [300,256]
    # in_3: [512,256], in_2: [512]
    out2a = torch.empty((300, 256), dtype=dtype, device=in_5.device)
    out2b = torch.empty((300, 256), dtype=dtype, device=in_5.device)

    M2, K2, N2 = 300, 256, 512
    grid2 = lambda META: (
        triton.cdiv(M2, META['BLOCK_M']),
        triton.cdiv(N2, META['BLOCK_N']),
    )

    fused_linear_k2_kernel[grid2](
        in_4.view(300, -1, 256).view(300, 256),
        in_3, in_2,
        out2a, out2b,
        M2, N2, K2,
        256, 256,        # stride_aka, stride_o2a (row stride of in4_flat, out2a)
        in_3.stride(0), in_3.stride(1),
        256, 256,        # stride_o2b  (row stride of out2b)
        DTYPE=DTYPE,
    )

    # ---- Reconstruct the original output tensor layout ----
    # in_4 -> tmp_9 -> tmp_10 [300,1,512] -> tmp_11 [300,1,256], tmp_12 [300,1,256]
    tmp_11 = out2a.view(300, 1, 256)   # [300,1,256]
    tmp_12 = out2b.view(300, 1, 256)   # [300,1,256]
    tmp_8  = out1b.view(300, 256)      # [300,256]
    # tmp_13 = tmp_6.unsqueeze(-2): tmp_6 = tmp_5.view(-1,256) = out1a[300,256] -> [1,300,256]
    tmp_13 = out1a.view(1, 300, 256)   # [1,300,256]

    return (tmp_11, tmp_12, tmp_8, tmp_13)


def replacement_func():
    return linear_split_fused