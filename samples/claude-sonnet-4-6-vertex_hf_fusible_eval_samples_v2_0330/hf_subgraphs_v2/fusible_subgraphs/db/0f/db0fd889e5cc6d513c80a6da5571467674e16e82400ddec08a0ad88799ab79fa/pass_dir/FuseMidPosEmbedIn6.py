"""
Pass 2: FuseMidPosEmbedIn6 (simpler single-output pattern)

Key insight:
  tmp_28 = in_6[:, :, 1:-10, :] has shape [4, 1, 225, 32].
  The sub-chain:
    tmp_28 → transpose(2,3) → view(4,32,15,15)
    → interpolate(size=(15,15), bicubic)   ← IDENTITY (same input/output size!)
    → flatten(2) → transpose(1,2) → contiguous() → view(4,1,225,32) = tmp_35

  is equivalent to tmp_28.contiguous().
  Replace with a Triton kernel that copies the non-contiguous slice to contiguous output.
"""

import torch
import triton
import triton.language as tl


def pattern(x):
    # x = tmp_28 = in_6[:, :, 1:-10, :], shape [4, 1, 225, 32]
    tmp_29 = x.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    tmp_31 = torch.nn.functional.interpolate(tmp_30, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_32 = tmp_31.flatten(2)
    tmp_33 = tmp_32.transpose(1, 2)
    tmp_34 = tmp_33.contiguous()
    tmp_35 = tmp_34.view(4, 1, 225, 32)
    return tmp_35


def replacement_args(x):
    return (x,)


@triton.jit
def copy_strided_4d_kernel(
    in_ptr,
    out_ptr,
    in_stride_b,
    in_stride_h,
    in_stride_l,
    B,
    H,
    L,
    D,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copies a [B, H, L, D] non-contiguous tensor (stride_d=1) to contiguous output.
    Used to replace the no-op transpose/view/interpolate/flatten/transpose/contiguous/view chain.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose linear output offset → (b, h, l, d)
    d_idx = offsets % D
    l_idx = (offsets // D) % L
    h_idx = (offsets // (D * L)) % H
    b_idx = offsets // (D * L * H)

    # Input offset uses the (non-contiguous) strides; stride_d=1 so d_idx is direct
    in_offsets = b_idx * in_stride_b + h_idx * in_stride_h + l_idx * in_stride_l + d_idx

    data = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offsets, data, mask=mask)


@torch.fx.wrap
def interpolate_noop_mid_patch(x):
    # x: [4, 1, 225, 32] with strides (7552, 7552, 32, 1) - non-contiguous slice of in_6
    # The roundtrip is identity; produce a contiguous [4, 1, 225, 32] output.
    B, H, L, D = 4, 1, 225, 32
    out = x.new_empty(B, H, L, D)   # same dtype/device as x, avoids torch.empty
    n_elements = B * H * L * D   # 28800
    s_b, s_h, s_l, _s_d = x.stride()

    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    copy_strided_4d_kernel[(num_programs,)](
        in_ptr=x,
        out_ptr=out,
        in_stride_b=s_b,
        in_stride_h=s_h,
        in_stride_l=s_l,
        B=B, H=H, L=L, D=D,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return interpolate_noop_mid_patch