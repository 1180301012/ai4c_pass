import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    in_0_ptr,       # scalar (scale)
    in_1_ptr,       # tensor to add
    in_2_ptr,       # tensor (modified in-place, used as base)
    in_3_ptr,       # tensor to add
    out_ptr,        # output tensor
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: (in_2 + in_1) * in_0 + in_3 -> contiguous output"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load in_1 (tensor to add)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Load in_2 (base tensor, modified in-place)
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    
    # Load in_3 (tensor to add)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Load in_0 (scalar scale)
    in_0 = tl.load(in_0_ptr)
    
    # Compute: (in_2 + in_1) * in_0 + in_3
    # Equivalent to: tmp = in_2 + in_1; tmp = tmp * in_0; result = tmp + in_3
    tmp = in_2 + in_1
    tmp = tmp * in_0
    result = tmp + in_3
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_op(in_0, in_1, in_2, in_3):
    """
    Fused operation: (in_2 + in_1) * in_0 + in_3 -> contiguous output
    
    This fuses 4 operations into 1:
    1. in_2 += in_1 (in-place add)
    2. tmp_2 = in_2 * in_0 (scalar multiply)
    3. tmp_3 = tmp_2 + in_3 (add)
    4. tmp_4 = tmp_3.contiguous() (make contiguous)
    """
    # Get total number of elements (assuming in_1, in_2, in_3 have same shape)
    n_elements = in_1.numel()
    
    # Choose block size - power of 2 for efficiency
    BLOCK_SIZE = 1024
    
    # Calculate grid
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor (contiguous)
    out = torch.empty_like(in_1)
    
    # Launch kernel
    fused_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the computation pattern:
    tmp_0 = in_0
    in_2 += in_1
    tmp_1 = in_2
    tmp_2 = tmp_1 * tmp_0
    tmp_1 = tmp_0 = None
    tmp_3 = tmp_2 + in_3
    tmp_2 = None
    tmp_4 = tmp_3.contiguous()
    tmp_3 = None
    return (tmp_4,)
    
    This can be simplified to:
    in_2 += in_1
    tmp_2 = in_2 * in_0
    tmp_3 = tmp_2 + in_3
    tmp_4 = tmp_3.contiguous()
    return (tmp_4,)
    """
    # Step 1: in-place add
    in_2 += in_1
    
    # Step 2: scalar multiply (in_0 is a scalar)
    tmp_2 = in_2 * in_0
    
    # Step 3: add
    tmp_3 = tmp_2 + in_3
    
    # Step 4: make contiguous
    tmp_4 = tmp_3.contiguous()
    
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_op