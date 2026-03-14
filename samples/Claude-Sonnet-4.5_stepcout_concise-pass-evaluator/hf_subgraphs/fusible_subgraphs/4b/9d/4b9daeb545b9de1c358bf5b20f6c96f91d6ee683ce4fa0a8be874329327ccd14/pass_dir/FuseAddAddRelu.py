import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    """
    Pattern to match: three additions followed by ReLU
    This mirrors the exact operations from model.py
    """
    in_0 += in_1
    tmp_0 = in_0
    tmp_0 += in_3
    tmp_1 = tmp_0
    tmp_2 = torch.nn.functional.relu(tmp_1, inplace=False)
    return tmp_2


def replacement_args(in_0, in_1, in_3):
    """Extract arguments needed for the replacement"""
    return (in_0, in_1, in_3)


@triton.jit
def fused_add_add_relu_kernel(
    in_0_ptr, in_1_ptr, in_3_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: out = relu(in_0 + in_1 + in_3)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Compute: (in_0 + in_1 + in_3) then ReLU
    result = in_0 + in_1 + in_3
    result = tl.maximum(result, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_add_relu(in_0, in_1, in_3):
    """
    Wrapper function to launch the fused kernel
    """
    # Output tensor
    out = torch.empty_like(in_0)
    
    # Get total number of elements
    n_elements = in_0.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_add_add_relu_kernel[grid](
        in_0, in_1, in_3, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """Return the replacement function"""
    return fused_add_add_relu