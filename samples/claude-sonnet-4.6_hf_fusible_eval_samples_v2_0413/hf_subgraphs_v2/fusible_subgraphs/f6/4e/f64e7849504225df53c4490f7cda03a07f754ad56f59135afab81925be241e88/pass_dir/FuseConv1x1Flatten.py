import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match: conv2d (1x1, stride=1, pad=0, dilation=1, groups=1)
#                   followed by flatten(result, 2)
# in_0 = bias  [C_out]
# in_1 = weight [C_out, C_in, 1, 1]
# in_2 = input  [N, C_in, H, W]
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused 1x1-conv + flatten
# Treats the problem as:  output[n, m, hw] = sum_k weight[m,k] * input[n,k,hw] + bias[m]
# Grid: (ceil(C_out/BM), ceil(HW/BN), N)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # BLOCK_M must be a power-of-2 >= 16 for tl.dot
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['N', 'C_out', 'C_in', 'HW'],
)
@triton.jit
def _conv1x1_flatten_kernel(
    input_ptr,   # [N, C_in, HW]  (C-contiguous)
    weight_ptr,  # [C_out, C_in]  (C-contiguous)
    bias_ptr,    # [C_out]
    output_ptr,  # [N, C_out, HW] (C-contiguous)
    N,
    C_out,
    C_in,
    HW,
    BLOCK_M: tl.constexpr,   # tile size over C_out
    BLOCK_N: tl.constexpr,   # tile size over HW
    BLOCK_K: tl.constexpr,   # tile size over C_in
):
    pid_m = tl.program_id(0)   # which C_out tile
    pid_n = tl.program_id(1)   # which HW tile
    pid_b = tl.program_id(2)   # which batch element

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Pre-load bias for this C_out tile  [BLOCK_M]
    bias_vals = tl.load(bias_ptr + offs_m,
                        mask=offs_m < C_out, other=0.0).to(tl.float32)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointer bases (cast to int64 to avoid overflow for large N/HW/C_in)
    batch_in_offset  = pid_b.to(tl.int64) * C_in * HW
    batch_out_offset = pid_b.to(tl.int64) * C_out * HW

    # Inner reduction loop over C_in
    for k in range(0, tl.cdiv(C_in, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load weight tile  [BLOCK_M, BLOCK_K]
        w_mask = (offs_m[:, None] < C_out) & (offs_k[None, :] < C_in)
        w = tl.load(
            weight_ptr + offs_m[:, None] * C_in + offs_k[None, :],
            mask=w_mask, other=0.0
        )

        # Load input tile  [BLOCK_K, BLOCK_N]
        in_mask = (offs_k[:, None] < C_in) & (offs_n[None, :] < HW)
        inp = tl.load(
            input_ptr + batch_in_offset
                      + offs_k[:, None] * HW
                      + offs_n[None, :],
            mask=in_mask, other=0.0
        )

        # Matrix multiply-accumulate
        acc += tl.dot(w, inp)

    # Add bias (broadcast over HW)
    acc = acc + bias_vals[:, None]

    # Store result
    out_mask = (offs_m[:, None] < C_out) & (offs_n[None, :] < HW)
    tl.store(
        output_ptr + batch_out_offset
                   + offs_m[:, None] * HW
                   + offs_n[None, :],
        acc,
        mask=out_mask
    )


# ---------------------------------------------------------------------------
# Wrapper callable returned by replacement_func
# ---------------------------------------------------------------------------
@torch.fx.wrap
def conv1x1_flatten(bias, weight, input_tensor):
    N, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    HW = H * W

    # Allocate output  [N, C_out, HW]
    # weight has shape [C_out, C_in, 1, 1]; stride(0)=C_in, stride(1)=1
    # so weight_ptr + c*C_in + k  accesses weight[c, k, 0, 0] correctly
    output = torch.empty((N, C_out, HW),
                         dtype=input_tensor.dtype,
                         device=input_tensor.device)

    grid = lambda meta: (
        triton.cdiv(C_out, meta['BLOCK_M']),
        triton.cdiv(HW,    meta['BLOCK_N']),
        N,
    )

    _conv1x1_flatten_kernel[grid](
        input_tensor, weight, bias, output,
        N, C_out, C_in, HW,
    )

    return output


def replacement_func():
    return conv1x1_flatten