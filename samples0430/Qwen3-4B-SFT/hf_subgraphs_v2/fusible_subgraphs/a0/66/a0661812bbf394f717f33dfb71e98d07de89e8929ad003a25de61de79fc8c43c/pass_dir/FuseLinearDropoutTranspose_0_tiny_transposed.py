"""
Pass: FuseLinearDropoutTranspose_0_tiny_transposed
Matches: F.linear + F.dropout(p=0.0, training=False) + transpose(1,2)
Returns: (transposed_result, linear_result)  -- transposed-first order
Target: float16 tiny model (S=1248, H=16, K=32)
"""
import torch
import triton
import triton.language as tl
from pass_dir.fused_matmul_transpose_kernel import _fused_linear_transpose_kernel


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return (tmp_4, tmp_3)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def _fused_linear_transpose_tiny(bias, weight, x):
    # Use shape-derived strides to avoid PoisonDispatchTensor dispatch restrictions.
    B  = x.shape[0]
    S  = x.shape[1]
    K  = x.shape[2]
    H  = weight.shape[0]
    M  = B * S
    N  = H
    stride_Xm = x.shape[1] * x.shape[2]   # contiguous [B,S,K]: stride(1)
    stride_wn = x.shape[2]                # contiguous [H,K]: stride(0) = K

    y      = torch.empty(x.shape[0], x.shape[1], H,         dtype=x.dtype, device=x.device)
    y_T    = torch.empty(H,          x.shape[1], x.shape[0], dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _fused_linear_transpose_kernel[grid](
        x, weight, bias, y, y_T,
        M, N, K,
        stride_Xm, 1,
        stride_wn, 1,
        N, M,
        DIRECT=False,
    )

    out_linear = y.view(B, S, H)   # Y  (second return value in pattern)
    out_transposed = y_T.view(B, H, S)  # Y^T  (first return value in pattern)
    return (out_transposed, out_linear)


def replacement_func():
    return _fused_linear_transpose_tiny