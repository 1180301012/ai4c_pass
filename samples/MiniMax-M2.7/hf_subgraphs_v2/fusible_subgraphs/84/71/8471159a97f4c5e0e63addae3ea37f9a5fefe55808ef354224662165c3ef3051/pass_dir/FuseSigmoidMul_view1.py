import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Match: linear -> sigmoid -> view(1, 64, 1, 1) -> mul pattern
    Returns view output and final multiplication result
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(1, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_4, tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def sigmoid_mul_kernel(output_ptr, in_3_ptr, sigmoid_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    sig = tl.load(sigmoid_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    out = sig * y
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_sigmoid_mul(linear_out, in_3):
    N = in_3.numel()
    BLOCK_SIZE = 256
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # First compute sigmoid of linear output
    sigmoid_out = torch.sigmoid(linear_out)
    # View to broadcast shape
    sigmoid_view = sigmoid_out.view(1, 64, 1, 1)
    # Use Triton kernel for the multiplication
    out = torch.empty_like(in_3)
    sigmoid_ptr = sigmoid_view.expand(in_3.shape).contiguous()
    
    sigmoid_mul_kernel[(num_programs,)](
        output_ptr=out,
        in_3_ptr=in_3,
        sigmoid_ptr=sigmoid_ptr,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_sigmoid_mul