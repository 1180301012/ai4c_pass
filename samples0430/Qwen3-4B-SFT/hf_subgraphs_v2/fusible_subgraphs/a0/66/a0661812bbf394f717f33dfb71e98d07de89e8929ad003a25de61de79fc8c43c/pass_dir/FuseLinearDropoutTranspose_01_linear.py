"""
Pass: FuseLinearDropoutTranspose_01
Matches: F.linear + F.dropout(p=0.1, training=False) + transpose(1,2)
Returns: (linear_result, transposed_result)  -- linear-first order
Target: float16 graph (Hubertemotion, S=249, H=768, K=512)
"""
import torch
import triton
import triton.language as tl
from pass_dir.fused_matmul_transpose_kernel import _fused_linear_transpose_kernel


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    # in_0=bias [H], in_1=weight [H,K], in_2=x [S,K]
    return (in_0, in_1, in_2)


@torch.fx.wrap
def _fused_linear_transpose_01(bias, weight, x):
    # Use shape-derived strides to avoid PoisonDispatchTensor dispatch restrictions.
    # For contiguous input: stride(m) = product of all dims after dim m.
    _shape = x.shape       # torch.Size – no dispatch
    _stride = weight.stride()._as_strided  # – no dispatch
    B = _shape[0]
    S = _shape[1]
    K = _shape[2]
    H = weight.shape[0]
    M = B * S
    N = H
    stride_Xm = _shape[1] * _shape[2]   # for contiguous [B,S,K] tensor: stride(1)

    y = torch.empty((_shape[0], _shape[1], H), dtype=x.dtype, device=x.device)
    y_T = torch.empty((H, _shape[1], _shape[0]), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _fused_linear_transpose_kernel[grid](
        x, weight, bias, y, y_T,
        M, N, K,
        stride_Xm, 1,    # stride_Xm, stride_Xk
        _stride[0], 1,   # stride_wn, stride_wk  (weight is [H, K], contiguous)
        N, M,           # stride_yrm (Y row-stride = H = N), stride_ytym (Y_T row-stride = M)
        DIRECT=False,
    )

    out_linear = y.view(B, S, H)
    out_transposed = y_T.view(B, H, S)
    return (out_linear, out_transposed)


def replacement_func():
    return _fused_linear_transpose_01