import torch
import triton
import triton.language as tl


def pattern(bias, weight, input_tensor):
    conv2d = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.autotune(
    configs=[
        # 8 configs total → ~24 autotune calls (within warmup=25 budget)
        # BLOCK_M=32, BLOCK_N=64: 20 blocks for M=32 — best GPU utilization
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8, num_stages=4),
        # BLOCK_M=32, BLOCK_N=128: 10 blocks, larger tile per block
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
        # BLOCK_M=16, BLOCK_N=128: 20 blocks
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=4),
    ],
    key=['M', 'C_in', 'C_out'],
)
@triton.jit
def fused_conv1x1_hardswish_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, C_in, C_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, C_in, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # evict_last: keep small input and large weight in L2 across N-blocks
        inp = tl.load(input_ptr + m_offs[:, None] * C_in + k_offs[None, :],
                      mask=(m_offs[:, None] < M) & (k_offs[None, :] < C_in),
                      other=0.0, eviction_policy="evict_last")
        weight = tl.load(weight_ptr + n_offs[:, None] * C_in + k_offs[None, :],
                         mask=(n_offs[:, None] < C_out) & (k_offs[None, :] < C_in),
                         other=0.0, eviction_policy="evict_last")
        acc += tl.dot(inp, tl.trans(weight), allow_tf32=True)
    bias = tl.load(bias_ptr + n_offs, mask=n_offs < C_out, other=0.0)
    acc += bias[None, :].to(tl.float32)
    shifted = acc + 3.0
    clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    acc = acc * clamped * (1.0 / 6.0)
    tl.store(output_ptr + m_offs[:, None] * C_out + n_offs[None, :],
             acc.to(output_ptr.dtype.element_ty),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < C_out))


@torch.fx.wrap
def fused_conv1x1_hardswish_flatten(bias, weight, input_tensor):
    N     = input_tensor.shape[0]
    C_in  = input_tensor.shape[1]
    C_out = weight.shape[0]
    output = torch.empty((N, C_out), dtype=input_tensor.dtype, device=input_tensor.device)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']),
                         triton.cdiv(C_out, meta['BLOCK_N']))
    fused_conv1x1_hardswish_kernel[grid](
        input_tensor, weight, bias, output,
        N, C_in, C_out,
    )
    return output


def replacement_func():
    return fused_conv1x1_hardswish_flatten