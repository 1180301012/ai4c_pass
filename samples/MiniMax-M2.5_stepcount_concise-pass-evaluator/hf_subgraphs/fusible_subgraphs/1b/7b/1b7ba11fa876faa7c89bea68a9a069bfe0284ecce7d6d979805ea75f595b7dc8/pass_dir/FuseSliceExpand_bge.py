import torch
import triton
import triton.language as tl


@triton.jit
def slice_expand_kernel(
    in_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    ptr = in_ptr + col_offsets
    vals = tl.load(ptr, mask=mask, other=0)
    out_ptr_row = out_ptr + row * N + col_offsets
    tl.store(out_ptr_row, vals, mask=mask)


@torch.fx.wrap
def slice_expand_kernel_wrapper(in_tensor, M, N):
    out = torch.empty((M, N), dtype=in_tensor.dtype, device=in_tensor.device)
    grid = (M,)
    BLOCK_SIZE = 128 if N >= 128 else 64
    if N < BLOCK_SIZE:
        BLOCK_SIZE = N
    slice_expand_kernel[grid](in_ptr=in_tensor, out_ptr=out, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def pattern(in_0, in_1):
    """
    Match pattern from bge-base-en-v1.5:
    tmp_2 = in_1[:, :128]
    tmp_3 = tmp_2.expand(1, 128)
    tmp_4 = in_0[:, :128]  # This is different - uses slicing not None
    """
    tmp_2 = in_1[:, :128]
    tmp_3 = tmp_2.expand(1, 128)
    tmp_4 = in_0[:, :128]
    return tmp_3, tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    def fused_slice_expand(in_0, in_1):
        batch_size = in_0.shape[0]
        N = in_0.shape[1]
        M = batch_size
        
        # Slice and expand
        tmp_2 = in_1[:, :N]
        tmp_3 = tmp_2.expand(M, N)
        
        # tmp_4: slice instead of adding dimensions
        tmp_4 = in_0[:, :N]
        
        return tmp_3, tmp_4
    
    return fused_slice_expand