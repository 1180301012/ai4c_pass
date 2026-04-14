import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Matches: linear(in_2, in_1, in_0) -> transpose(-1,-2) -> multiply by in_3
    in_0: bias  [M]
    in_1: weight [M, K]
    in_2: input  [B, N, K]
    in_3: gate   [B, M, N]
    output: [B, M, N]
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        # Large blocks — best throughput for big batches (B=128) fp16/bf16/fp32
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=8),
        # Medium blocks — balanced for B=32 and B=128 fp32
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        # Small blocks — more thread blocks for B=1 occupancy
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['B', 'N', 'M', 'K', 'USE_FP16'],
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    in2_ptr, in1_ptr, in0_ptr, in3_ptr, out_ptr,
    B, N, M, K,
    stride_in2_b, stride_in2_n, stride_in2_k,
    stride_in1_m, stride_in1_k,
    stride_in3_b, stride_in3_m, stride_in3_n,
    stride_out_b, stride_out_m, stride_out_n,
    USE_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute: out[b, m, n] = in3[b, m, n] * (sum_k in2[b, n, k] * in1[m, k] + in0[m])
    Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    Uses tl.make_block_ptr + tl.advance for efficient pipelined 2D block loads.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Block pointers for efficient pipelined loads with boundary checking
    # in2[b, n_start:n_start+BN, 0:K]  — shape (N, K), access (BN, BK) tile
    in2_block_ptr = tl.make_block_ptr(
        base=in2_ptr + pid_b * stride_in2_b,
        shape=(N, K),
        strides=(stride_in2_n, stride_in2_k),
        offsets=(n_start, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    # in1[m_start:m_start+BM, 0:K]  — shape (M, K), access (BM, BK) tile
    in1_block_ptr = tl.make_block_ptr(
        base=in1_ptr,
        shape=(M, K),
        strides=(stride_in1_m, stride_in1_k),
        offsets=(m_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )

    # Float32 accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        # Pipelined block loads with hardware boundary checking & zero-padding
        in2_block = tl.load(in2_block_ptr, boundary_check=(0, 1), padding_option='zero')
        in1_block = tl.load(in1_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # acc += in1[BM,BK] @ in2.T[BK,BN]
        # fp16/bf16 uses fp16 TC; fp32 uses TF32 TC via allow_tf32=True
        acc = tl.dot(in1_block, tl.trans(in2_block), acc, allow_tf32=True)

        # Advance block pointers along K dimension
        in2_block_ptr = tl.advance(in2_block_ptr, (0, BLOCK_K))
        in1_block_ptr = tl.advance(in1_block_ptr, (0, BLOCK_K))

    # Index ranges for bias and elementwise ops
    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)

    # Add bias in0[m]: broadcast [BLOCK_M] -> [BLOCK_M, BLOCK_N]
    bias = tl.load(in0_ptr + m_range, mask=m_range < M, other=0.0).to(tl.float32)
    acc = acc + bias[:, None]

    # Load in3[b, m, n] and element-wise multiply
    in3_ptrs = (in3_ptr
                + pid_b * stride_in3_b
                + m_range[:, None] * stride_in3_m
                + n_range[None, :] * stride_in3_n)
    mask_out = (m_range[:, None] < M) & (n_range[None, :] < N)
    in3_block = tl.load(in3_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    result = acc * in3_block

    # Store with correct output dtype
    out_ptrs = (out_ptr
                + pid_b * stride_out_b
                + m_range[:, None] * stride_out_m
                + n_range[None, :] * stride_out_n)
    if USE_FP16 == 1:
        tl.store(out_ptrs, result.to(tl.float16), mask=mask_out)
    elif USE_FP16 == 2:
        tl.store(out_ptrs, result.to(tl.bfloat16), mask=mask_out)
    else:
        tl.store(out_ptrs, result, mask=mask_out)


@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    Fused: linear(in_2, in_1, in_0).transpose(-1,-2) * in_3

    in_0 : bias    [M=196]
    in_1 : weight  [M=196, K=196]
    in_2 : input   [B, N=768, K=196]
    in_3 : gate    [B, M=196, N=768]
    out  :         [B, M=196, N=768]
    """
    B = in_2.shape[0]
    N = in_2.shape[1]   # 768
    K = in_2.shape[2]   # 196
    M = in_1.shape[0]   # 196

    out = torch.empty_like(in_3)

    _dtype_map = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
    use_fp16 = _dtype_map.get(in_3.dtype, 0)

    grid = lambda meta: (
        B,
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, N, M, K,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        in_1.stride(0), in_1.stride(1),
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        USE_FP16=use_fp16,
    )

    return out


def replacement_func():
    return fused_linear_transpose_mul