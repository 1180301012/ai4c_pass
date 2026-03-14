import torch
import triton
import triton.language as tl


# ============================================================
# PATTERN: Graph 0 (efficient_mlm) - slice(64), expand(1, 64), in_0[:, None, None, :]
# ============================================================

@triton.jit
def slice_expand_kernel_1_64(
    in_1_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for slice and expand with N=64, B=1"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(in_1_ptr + offsets, mask=mask, other=0)
    
    # B=1, so just store once
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_fused_1_64(in_0, in_1):
    """Fused operation for graph 0: slice [:64], expand (1, 64)"""
    # Use literal values instead of variables to avoid FX tracing issues
    out = torch.empty((1, 64), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 64
    num_programs = 1  # (64 + 64 - 1) // 64 = 1
    
    in_1_contiguous = in_1.contiguous()
    
    slice_expand_kernel_1_64[(num_programs,)](
        in_1_ptr=in_1_contiguous,
        out_ptr=out,
        N=64,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    in_0_unsqueezed = in_0[:, None, None, :]
    return out, in_0_unsqueezed


def pattern(in_0, in_1):
    """Pattern for Graph 0 (efficient_mlm): slice(64), expand(1, 64)"""
    # Use explicit slice objects to match the model exactly
    tmp_2 = in_1[slice(None, None, None), slice(None, 64, None)]
    tmp_3 = tmp_2.expand(1, 64)
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    return tmp_3, tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_fused_1_64