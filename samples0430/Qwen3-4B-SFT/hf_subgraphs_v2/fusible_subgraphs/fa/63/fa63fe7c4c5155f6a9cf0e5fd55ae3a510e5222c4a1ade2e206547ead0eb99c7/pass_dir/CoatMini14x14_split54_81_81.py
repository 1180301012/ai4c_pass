"""
Pass for: in_1 @ in_0 + in_1[:,:,1:,:] + in_2[:,:,1:,:].T.reshape(1,216,14,14).split([54,81,81],1)
Matches: coat_mini (float16/bf16/float32), reshape(1,216,14,14), split [54,81,81]
"""
import torch
from pass_dir.kernel_impl import dispatch_func


def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 216, 14, 14)
    tmp_5 = torch.functional.split(tmp_4, [54, 81, 81], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return (tmp_0, tmp_6, tmp_7, tmp_8, tmp_1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "p3")


@torch.fx.wrap
def _dispatch_matmul_transpose_split(in_1, in_0, in_2, route):
    dim_h   = in_2.shape[1]
    dim_t   = in_2.shape[2]
    dim_d   = in_2.shape[3]
    T       = dim_t
    D       = dim_d
    N_T_out = dim_h * T
    C_out   = dim_h * D
    H_out   = 14
    L_out   = 14

    out0 = torch.empty((1,  54, 14, 14), dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty((1,  81, 14, 14), dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((1,  81, 14, 14), dtype=in_1.dtype, device=in_1.device)
    out2_new = torch.empty((1,  81, 14, 14), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        triton.cdiv(N_T_out, meta['BLOCK_M']),
        dim_h,
        triton.cdiv(14 * 14, meta['BLOCK_M']),
    )
    fused_matmul_transpose_split_kernel[grid](
        in_1, in_0, in_2,
        out0, out1, out2,
        dim_h, T, D,
        C_out, H_out, L_out, dim_h,
    )
    return out0, out1, out2

def replacement_func():
    return dispatch_func