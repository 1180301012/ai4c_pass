import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Fused 1×1 conv2d + flatten as a batched GEMM.
#
# Grid: (ceil(HW/BLOCK_N), ceil(C_out/BLOCK_M), N)
#
# Optimisations:
#   • Native dtype in tl.dot → tensor-core (HMMA fp16/bf16, TF32 fp32).
#   • evict_last on weight loads  → keep weight tiles in L1 cache across K.
#   • evict_first on input loads  → don't pollute L1 with single-use data.
#   • fp32 accumulator for numerical parity with cuDNN.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # 6 targeted configs to minimize autotuning overhead and IQR instability
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
    ],
    key=['C_out', 'HW', 'C_in'],
    warmup=5,
    rep=10,
)
@triton.jit
def conv1x1_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C_out, HW, C_in,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_n  = tl.program_id(2)

    co_offs = pid_co * BLOCK_M + tl.arange(0, BLOCK_M)
    hw_offs = pid_hw * BLOCK_N + tl.arange(0, BLOCK_N)
    co_mask = co_offs < C_out
    hw_mask = hw_offs < HW

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_in

        # Weight: small and reused → keep in L1 (evict_last)
        w_offs = co_offs[:, None] * C_in + k_offs[None, :]
        w_tile = tl.load(weight_ptr + w_offs,
                         mask=co_mask[:, None] & k_mask[None, :],
                         other=0.0,
                         eviction_policy='evict_last')

        # Input: large, single-use → don't pollute L1 (evict_first)
        i_base = pid_n * C_in * HW
        i_offs = i_base + k_offs[:, None] * HW + hw_offs[None, :]
        i_tile = tl.load(input_ptr + i_offs,
                         mask=k_mask[:, None] & hw_mask[None, :],
                         other=0.0,
                         eviction_policy='evict_first')

        # Native dtype → tensor-core (HMMA for fp16/bf16, TF32 for fp32)
        acc += tl.dot(w_tile, i_tile, out_dtype=tl.float32)

    bias_val = tl.load(bias_ptr + co_offs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias_val[:, None]

    out_base = pid_n * C_out * HW
    out_offs = out_base + co_offs[:, None] * HW + hw_offs[None, :]
    tl.store(output_ptr + out_offs,
             acc.to(OUTPUT_DTYPE),
             mask=co_mask[:, None] & hw_mask[None, :])


@torch.fx.wrap
def conv1x1_flatten_triton(bias, weight, x):
    N     = x.shape[0]
    C_in  = x.shape[1]
    H     = x.shape[2]
    W     = x.shape[3]
    C_out = weight.shape[0]
    HW    = H * W

    output = torch.empty((N, C_out, HW), dtype=x.dtype, device=x.device)

    _dtype_map = {
        torch.float32:  tl.float32,
        torch.float16:  tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    output_dtype = _dtype_map[x.dtype]

    grid = lambda meta: (
        triton.cdiv(HW,    meta['BLOCK_N']),
        triton.cdiv(C_out, meta['BLOCK_M']),
        N,
    )

    conv1x1_flatten_kernel[grid](
        x, weight, bias, output,
        C_out, HW, C_in,
        OUTPUT_DTYPE=output_dtype,
    )

    return output


def replacement_func():
    return conv1x1_flatten_triton