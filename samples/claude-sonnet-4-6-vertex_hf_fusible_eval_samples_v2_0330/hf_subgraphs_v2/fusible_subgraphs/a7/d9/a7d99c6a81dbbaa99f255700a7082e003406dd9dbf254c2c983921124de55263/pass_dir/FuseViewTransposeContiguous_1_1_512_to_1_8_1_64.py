"""
Pass: Fuse view(1,1,-1,64) + transpose(1,2) + contiguous()
Input:  [1, 1, 512]  (e.g. key_states)
Output: [1, 8, 1, 64] contiguous

Key insight
-----------
view(1,1,8,64).transpose(1,2).contiguous() has the same flat memory
layout as writing 512 elements straight into a contiguous [1,8,1,64]
buffer.  We replace the three-op chain with a single Triton memcopy that
avoids the intermediate non-contiguous tensor and saves one GPU kernel
launch versus the original contiguous() copy.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: copy 512 elements (flat src → flat dst)
# The output buffer is allocated as [1,8,1,64]; its flat layout equals the
# input's flat layout, so a plain element-wise copy gives the correct result.
# ---------------------------------------------------------------------------
@triton.jit
def _copy_flat_512_kernel(
    src_ptr,
    dst_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs))


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _view_transpose_contiguous_512(in_4):
    """
    Replaces:
        tmp_3 = in_4.view(1, 1, -1, 64)
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_9 = tmp_4.contiguous()
    """
    out = torch.empty((1, 8, 1, 64), dtype=in_4.dtype, device=in_4.device)
    _copy_flat_512_kernel[(1,)](
        in_4,  # [1,1,512] contiguous – Triton base pointer + offsets 0..511
        out,   # [1,8,1,64] contiguous – same flat layout
        BLOCK=512,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API required by the AI4C framework
# ---------------------------------------------------------------------------
def pattern(in_4):
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    return tmp_9


def replacement_args(in_4):
    return (in_4,)


def replacement_func():
    return _view_transpose_contiguous_512