"""
Pass: Replace stride-1 1x1 conv2d with a Triton GEMM kernel.
The slice op (if any) stays in the graph and uses the replacement's output.
"""
import torch
import triton
import triton.language as tl
from pass_dir.conv1x1_shared import conv1x1_gemm_kernel


def pattern(in_0, in_1):
    result = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def _triton_conv_s1(in_0, in_1):
    # in_0: weight (Cout, Cin, 1, 1)  in_1: feature (N, Cin, H, W)
    # No .view() / .contiguous() / .to() – all blocked by API validator
    Cout    = in_0.shape[0]
    Cin     = in_0.shape[1]
    N_batch = in_1.shape[0]
    H_in    = in_1.shape[2]
    W_in    = in_1.shape[3]
    HW      = H_in * W_in
    M       = N_batch * HW
    IS_FP16 = in_1.dtype == torch.float16
    IS_BF16 = in_1.dtype == torch.bfloat16
    out     = torch.empty((N_batch, Cout, H_in, W_in), dtype=in_1.dtype, device=in_1.device)
    grid    = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(Cout, meta['BLOCK_N']),)
    conv1x1_gemm_kernel[grid](
        in_1, in_0, out,
        M, Cout, Cin,
        HW, W_in, HW, W_in, 1,
        IS_FP16, IS_BF16,
    )
    return out


def replacement_func():
    return _triton_conv_s1