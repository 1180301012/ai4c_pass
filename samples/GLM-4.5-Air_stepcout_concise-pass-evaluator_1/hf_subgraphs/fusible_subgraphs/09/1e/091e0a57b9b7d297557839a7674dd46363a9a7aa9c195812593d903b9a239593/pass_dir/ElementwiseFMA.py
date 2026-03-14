import torch
import triton
import triton.language as tl

@triton.jit
def fma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    total_elements = N * C * H * W
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Fused multiply-add: a * b + c
    result = a * b + c
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fma_operation(a, b, c):
    N, C, H, W = a.shape
    total_elements = N * C * H * W
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(a)
    
    fma_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        output_ptr=output,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(sigmoid_input, x1, x0):
    """Match the computed sigmoid then element-wise multiply-add"""
    tmp_0 = sigmoid_input.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(x1)
    tmp_3 = x1 * tmp_2
    tmp_3 = tmp_3 + x0
    return tmp_3

def replacement_args(sigmoid_input, x1, x0):
    return (sigmoid_input, x1, x0)

def replacement_func():
    return fma_operation