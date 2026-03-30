import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1×1 conv2d + flatten(dim=2)
# Equivalent to batched GEMM:  output[n,cout,hw] = Σ_cin weight[cout,cin]*inp[n,cin,hw] + bias[cout]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    """
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : input  [N, C_in, H, W]
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Single Triton kernel: 3-D batched GEMM via tl.dot (tensor cores).
#
# Grid: (N, ceil(C_out/BLOCK_M), ceil(HW/BLOCK_N))
#
# Memory layout (input NCHW treated as [N, C_in, HW]):
#   input  [N, C_in, HW]  strides (C_in*HW, HW, 1)
#   weight [C_out, C_in]  strides (C_in, 1)
#   output [N, C_out, HW] strides (C_out*HW, HW, 1)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Small tiles – more programs, better SM utilisation for tiny N
        triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        # Medium tiles – balanced for mid-range N
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        # Large tiles – fewer programs, better throughput for large N
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C_out', 'C_in', 'HW'],
)
@triton.jit
def _conv1x1_flatten_kernel(
    inp_ptr, wgt_ptr, bias_ptr, out_ptr,
    N, C_in, C_out, HW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pid_m    = tl.program_id(1)
    pid_n    = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    inp_base = inp_ptr + batch_id * C_in * HW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(C_in, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        w_ptrs = wgt_ptr + offs_m[:, None] * C_in + offs_k[None, :]
        w_mask = (offs_m[:, None] < C_out) & (offs_k[None, :] < C_in)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        x_ptrs = inp_base + offs_k[:, None] * HW + offs_n[None, :]
        x_mask = (offs_k[:, None] < C_in) & (offs_n[None, :] < HW)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w, x, out_dtype=tl.float32)

    bias = tl.load(bias_ptr + offs_m, mask=offs_m < C_out, other=0.0)
    acc  = acc + bias[:, None].to(tl.float32)

    out_base = out_ptr + batch_id * C_out * HW
    out_ptrs = out_base + offs_m[:, None] * HW + offs_n[None, :]
    out_mask = (offs_m[:, None] < C_out) & (offs_n[None, :] < HW)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Wrapper callable (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def conv1x1_flatten_triton(bias, weight, inp):
    """
    bias   : [C_out]
    weight : [C_out, C_in, 1, 1]
    inp    : [N, C_in, H, W]
    returns: [N, C_out, H*W]
    """
    N, C_in, H, W = inp.shape
    C_out = weight.shape[0]
    HW    = H * W

    weight_2d = weight.reshape(C_out, C_in)
    output    = torch.empty((N, C_out, HW), dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        N,
        triton.cdiv(C_out, meta['BLOCK_M']),
        triton.cdiv(HW,    meta['BLOCK_N']),
    )

    _conv1x1_flatten_kernel[grid](
        inp, weight_2d, bias, output,
        N, C_in, C_out, HW,
    )
    return output


# ---------------------------------------------------------------------------
# replacement_func: return the wrapper (NOT a call)
# ---------------------------------------------------------------------------

def replacement_func():
    return conv1x1_flatten_triton