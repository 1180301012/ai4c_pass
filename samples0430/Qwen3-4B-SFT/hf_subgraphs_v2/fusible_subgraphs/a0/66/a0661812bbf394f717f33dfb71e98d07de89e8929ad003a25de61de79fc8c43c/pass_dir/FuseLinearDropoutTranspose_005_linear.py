"""
Pass: FuseLinearDropoutTranspose_005_linear
Matches: F.linear + F.dropout(p=0.05, training=False) + transpose(1,2)
Returns: (linear_result, transposed_result)  -- linear-first order
Target: bfloat16 graph (Unispeech-sat-russian-resd, S=249, H=1024, K=512)
"""
import torch
import triton
import triton.language as tl
from pass_dir.fused_matmul_transpose_kernel import _fused_linear_transpose_kernel


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.05, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def _fused_linear_transpose_005(bias, weight, x):
    # Use shape-derived strides to avoid PoisonDispatchTensor dispatch restrictions.
    B  = x.shape[0]
    S  = x.shape[1]
    K  = x.shape[2]
    H  = weight.shape[0]
    M  = B * S
    N  = H
    # Contiguous [B,S,K]: stride(1)=K, stride(2)=1
    stride_Xm = x.shape[1] * x.shape[2]
    # Contiguous [H,K]: stride(0)=K
    stride_wn = x.shape[2]

    y     = torch.empty(x.shape[0], x.shape[1], H,    dtype=x.dtype, device=x.device)
    y_T   = torch.empty(H,         x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _fused_linear_transpose_kernel[grid](
        x, weight, bias, y, y_T,
        M, N, K,
        stride_Xm, 1,
        stride_wn, 1,
        N, M,
        DIRECT=False,
    )

    out_linear   = y.view(B, S, H)
    out_transposed = y_T.view(B, H, S)
    return (out_linear, out_transposed)


def replacement_func():
    return _fused_linear_transpose_005