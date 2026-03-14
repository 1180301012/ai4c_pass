import torch
import triton
import triton.language as tl

# Pattern: flatten(2) followed by transpose(1, 2)
# Input: [B, C, H, W] -> flatten(2) -> [B, C, H*W] -> transpose(1,2) -> [B, H*W, C]
def pattern(x):
    t = x.flatten(2)
    out = t.transpose(1, 2)
    return out

def replacement_args(x):
    return (x,)

# Highly optimized kernel with better memory access patterns
@triton.jit
def flatten_transpose_fast_kernel(
    in_ptr, out_ptr,
    stride_b, stride_c, stride_hw,
    out_stride_b, out_stride_hw, out_stride_c,
    B, C, HW, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Output is [B, HW, C] contiguous
    b = offsets // (HW * C)
    temp = offsets % (HW * C)
    hw = temp // C
    c = temp % C
    
    # Input is [B, C, HW] contiguous (after x.contiguous())
    in_idx = b * stride_b + c * stride_c + hw * stride_hw
    
    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap  
def flatten_transpose_optimized(x):
    B, C, H, W = x.shape
    HW = H * W
    
    # For small tensors, just use PyTorch native (faster due to view semantics)
    total_elements = B * C * HW
    
    # Threshold: for very small tensors, native PyTorch is faster
    if total_elements < 65536:  # 64K elements
        return x.flatten(2).transpose(1, 2).contiguous()
    
    # For larger tensors, use our optimized kernel
    x_contig = x.contiguous()
    out = torch.empty((B, HW, C), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    flatten_transpose_fast_kernel[(num_programs,)](
        x_contig, out,
        C * HW, HW, 1,  # input strides for [B, C, HW]
        HW * C, C, 1,   # output strides for [B, HW, C]
        B, C, HW, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return flatten_transpose_optimized