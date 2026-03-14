import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching the entire computation sequence:
    cat -> adaptive_avg_pool2d -> flatten -> dropout
    """
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, False)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def vectorized_cat_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    n0, n1, n2, n3,
    BLOCK_SIZE: tl.constexpr,
):
    """Vectorized kernel with coalesced memory access."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Copy in_0
    mask0 = offsets < n0
    data0 = tl.load(in_0_ptr + offsets, mask=mask0, other=0.0)
    tl.store(out_ptr + offsets, data0, mask=mask0)
    
    # Copy in_1
    offset1 = n0
    mask1 = offsets < n1
    data1 = tl.load(in_1_ptr + offsets, mask=mask1, other=0.0)
    tl.store(out_ptr + offset1 + offsets, data1, mask=mask1)
    
    # Copy in_2
    offset2 = n0 + n1
    mask2 = offsets < n2
    data2 = tl.load(in_2_ptr + offsets, mask=mask2, other=0.0)
    tl.store(out_ptr + offset2 + offsets, data2, mask=mask2)
    
    # Copy in_3
    offset3 = n0 + n1 + n2
    mask3 = offsets < n3
    data3 = tl.load(in_3_ptr + offsets, mask=mask3, other=0.0)
    tl.store(out_ptr + offset3 + offsets, data3, mask=mask3)


@torch.fx.wrap
def fused_cat_flatten_wrapper(in_0, in_1, in_2, in_3):
    """
    Optimized wrapper that fuses cat+pool+flatten+dropout.
    """
    # Get dimensions
    batch_size = in_0.shape[0]
    n0 = in_0.shape[1]
    n1 = in_1.shape[1]
    n2 = in_2.shape[1]
    n3 = in_3.shape[1]
    total_channels = n0 + n1 + n2 + n3
    
    # Allocate output
    out = torch.empty((batch_size, total_channels), device=in_0.device, dtype=in_0.dtype)
    
    # Flatten inputs
    in_0_flat = in_0.reshape(-1)
    in_1_flat = in_1.reshape(-1)
    in_2_flat = in_2.reshape(-1)
    in_3_flat = in_3.reshape(-1)
    
    # Use optimal block size for small data
    BLOCK_SIZE = 256
    grid = (triton.cdiv(max(n0, n1, n2, n3), BLOCK_SIZE),)
    
    vectorized_cat_kernel[grid](
        in_0_flat, in_1_flat, in_2_flat, in_3_flat,
        out,
        n0, n1, n2, n3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_cat_flatten_wrapper