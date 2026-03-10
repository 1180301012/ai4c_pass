import torch
import triton
import triton.language as tl

def pattern(in_3, in_4, in_5, in_6, in_7):
    """
    Pattern: Multiple independent sigmoid operations
    This matches the 5 separate sigmoid calls on different inputs
    """
    tmp_4 = torch.nn.functional.sigmoid(in_3)
    tmp_5 = torch.nn.functional.sigmoid(in_4)
    tmp_6 = torch.nn.functional.sigmoid(in_5)
    tmp_7 = torch.nn.functional.sigmoid(in_6)
    tmp_8 = torch.nn.functional.sigmoid(in_7)
    return tmp_4, tmp_5, tmp_6, tmp_7, tmp_8

def replacement_args(in_3, in_4, in_5, in_6, in_7):
    """
    Extract arguments for the fused sigmoid kernel
    """
    return (in_3, in_4, in_5, in_6, in_7)

@triton.jit
def multi_sigmoid_kernel(
    in_ptr_0, in_ptr_1, in_ptr_2, in_ptr_3, in_ptr_4,
    out_ptr_0, out_ptr_1, out_ptr_2, out_ptr_3, out_ptr_4,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that applies sigmoid to multiple input tensors
    """
    # Each program processes a block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all 5 input tensors
    x0 = tl.load(in_ptr_0 + offsets, mask=mask, other=0.0)
    x1 = tl.load(in_ptr_1 + offsets, mask=mask, other=0.0)
    x2 = tl.load(in_ptr_2 + offsets, mask=mask, other=0.0)
    x3 = tl.load(in_ptr_3 + offsets, mask=mask, other=0.0)
    x4 = tl.load(in_ptr_4 + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid to all tensors in parallel
    sigmoid_0 = 1.0 / (1.0 + tl.exp(-x0))
    sigmoid_1 = 1.0 / (1.0 + tl.exp(-x1))
    sigmoid_2 = 1.0 / (1.0 + tl.exp(-x2))
    sigmoid_3 = 1.0 / (1.0 + tl.exp(-x3))
    sigmoid_4 = 1.0 / (1.0 + tl.exp(-x4))
    
    # Store all 5 results
    tl.store(out_ptr_0 + offsets, sigmoid_0, mask=mask)
    tl.store(out_ptr_1 + offsets, sigmoid_1, mask=mask)
    tl.store(out_ptr_2 + offsets, sigmoid_2, mask=mask)
    tl.store(out_ptr_3 + offsets, sigmoid_3, mask=mask)
    tl.store(out_ptr_4 + offsets, sigmoid_4, mask=mask)

@torch.fx.wrap
def fused_multi_sigmoid(in_3, in_4, in_5, in_6, in_7):
    """
    Wrapper function for the fused sigmoid kernel
    """
    # Determine the tensor size (assume all tensors have same size)
    n_elements = in_3.numel()
    
    # Create output tensors
    out_0 = torch.empty_like(in_3)
    out_1 = torch.empty_like(in_4)
    out_2 = torch.empty_like(in_5)
    out_3 = torch.empty_like(in_6)
    out_4 = torch.empty_like(in_7)
    
    # Set block size and grid configuration
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the fused kernel
    grid = (num_programs,)
    
    multi_sigmoid_kernel[grid](
        in_ptr_0=in_3,
        in_ptr_1=in_4,
        in_ptr_2=in_5,
        in_ptr_3=in_6,
        in_ptr_4=in_7,
        out_ptr_0=out_0,
        out_ptr_1=out_1,
        out_ptr_2=out_2,
        out_ptr_3=out_3,
        out_ptr_4=out_4,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_0, out_1, out_2, out_3, out_4

def replacement_func():
    """
    Return the fused sigmoid function
    """
    return fused_multi_sigmoid