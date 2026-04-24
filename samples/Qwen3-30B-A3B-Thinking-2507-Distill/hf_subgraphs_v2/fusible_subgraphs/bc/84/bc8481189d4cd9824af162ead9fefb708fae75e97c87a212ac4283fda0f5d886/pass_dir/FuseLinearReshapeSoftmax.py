import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: linear -> reshape -> softmax
# in_0: bias  [18]
# in_1: weight [18, 128]
# in_2: input  [1, 19, 128]
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel
# Grid: (B*M * 2,)  — one program per output row (38 total for this graph)
# Each program:
#   1. Accumulates dot(x[m, :], W[c, :]) over K=128 elements for all c in [0,18)
#   2. Adds bias
#   3. Computes softmax over 9 elements in its row
# Output layout: [B*M*2, 9]  = [38, 9]  (contiguous; tl.store treats as flat)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 128}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_K': 64},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_K': 32},  num_warps=2, num_stages=1),
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_K': 32},  num_warps=4, num_stages=1),
    ],
    key=['K', 'C'],
)
@triton.jit
def fused_linear_softmax_kernel(
    bias_ptr,    # [C=18]
    weight_ptr,  # [C=18, K=128]
    input_ptr,   # [B=1, M=19, K=128]  (contiguous, B*M rows of K)
    output_ptr,  # [B*M*2=38, C=9]     (contiguous, flat index = row*C + c)
    M,           # 19
    K,           # 128
    C,           # 18
    BLOCK_K: tl.constexpr,  # tuned
    BLOCK_C: tl.constexpr,  # 32  (>= C=18, next power of 2)
):
    row = tl.program_id(0)           # 0 .. 37
    group = row % 2                  # which of the two 9-element groups
    m = row // 2                     # row index in the original M=19 dimension

    # --- index vectors ---
    c_offs = tl.arange(0, BLOCK_C)   # [0 .. 31]
    c_mask = c_offs < C              # first 18 are valid

    # --- load bias once ---
    bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

    # --- accumulate GEMM: acc[c] = sum_k input[m,k] * weight[c,k] ---
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    x_base = m * K
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # input[m, k_offs]  — shape [BLOCK_K]
        x = tl.load(input_ptr + x_base + k_offs,
                     mask=k_mask, other=0.0).to(tl.float32)

        # weight[c_offs, k_offs]  — shape [BLOCK_C, BLOCK_K]
        w_ptrs = weight_ptr + c_offs[:, None] * K + k_offs[None, :]
        w_mask  = c_mask[:, None] & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # outer-product accumulate
        acc += tl.sum(w * x[None, :], axis=1)

    # --- add bias ---
    acc = acc + bias

    # --- softmax over ALL C=18 values per row (PyTorch's softmax(dim=1) covers all 18) ---
    # Output layout: [38, 9] flat — 2 programs per original row, each stores 9 values.
    # row*9 is the start of row-th output row (9 values per row, 38 rows = 342 elements).
    row_start  = row * 9
    exp_vals   = tl.where(c_offs < C, tl.exp(acc), 0.0)
    exp_sum    = tl.sum(exp_vals)
    softmax_out = exp_vals / exp_sum

    tl.store(output_ptr + row_start + c_offs,
             softmax_out.to(output_ptr.dtype.element_ty),
             mask=c_offs < C)  # c_offs=[0..31], mask keeps first C=18 positions


# ---------------------------------------------------------------------------
# Kernel wrapper (opaque to FX)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_linear_softmax(in_0, in_1, in_2):
    # in_0: bias   [18]
    # in_1: weight [18, 128]
    # in_2: input  [1, 19, 128]
    B, M, K = in_2.shape
    C = in_1.shape[0]   # 18

    # 2 groups per row; total rows = B*M*2 = 38
    total_rows = B * M * 2
    # output shape [38, 9] = [B*M*2, 9] = [342 elements]
    output = torch.empty((total_rows, 9), dtype=in_2.dtype, device=in_2.device)

    BLOCK_C = 32  # next power-of-2 >= C=18

    fused_linear_softmax_kernel[(total_rows,)](
        in_0, in_1, in_2, output,
        M, K, C,
        BLOCK_C=BLOCK_C,
    )

    return output.view(B * M * 2, 9, 1)  # [38, 9, 1] — 342 contiguous elements


def replacement_func():
    return fused_linear_softmax