import torch
import triton
import triton.language as tl

# Import the shared kernel (same dimensions, just different B wrapper)
from pass_dir.FuseReshapeSplitPermute import _fused_kernel

# Kernel wrapper specific to B=4
@torch.fx.wrap
def _fused_wrapper_b4(linear_out):
    B, S, C = linear_out.shape   # [4, 49, 1536]
    H      = 8
    Q   = torch.empty(B, H, S, 32,    dtype=linear_out.dtype, device=linear_out.device)
    K_T = torch.empty(B, H, 32,  S,    dtype=linear_out.dtype, device=linear_out.device)
    V   = torch.empty(B, H, 128, S,    dtype=linear_out.dtype, device=linear_out.device)
    _fused_kernel[(B * H,)](
        linear_out, Q, K_T, V,
        B, S, C,
        linear_out.stride(0), linear_out.stride(1), linear_out.stride(2),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_T.stride(0), K_T.stride(1), K_T.stride(2), K_T.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
    )
    return Q, K_T, V


def pattern(linear):
    x4d     = linear.reshape(4, 49, 8, -1)
    s       = x4d.split([32, 32, 128], dim=3)
    q   = s[0];  kt = s[1];  v = s[2]
    q1  = q.permute(0, 2, 1, 3)
    kt1 = kt.permute(0, 2, 1, 3)
    v1  = v.permute(0, 2, 1, 3)
    kt2 = kt1.transpose(-2, -1)
    return q1, kt2, v1


def replacement_args(linear):
    return (linear,)


def replacement_func():
    return _fused_wrapper_b4