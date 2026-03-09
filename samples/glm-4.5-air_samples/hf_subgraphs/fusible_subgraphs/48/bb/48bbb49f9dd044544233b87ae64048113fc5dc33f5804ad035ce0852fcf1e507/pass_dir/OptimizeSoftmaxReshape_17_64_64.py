import torch
import triton
import triton.language as tl

def pattern(softmax_input):
    tmp_2 = torch.nn.functional.softmax(softmax_input, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    return tmp_3

def replacement_args(softmax_input):
    return (softmax_input,)

@triton.jit
def softmax_reshape_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * (C * H * W)
    
    for i in range(0, C * H * W, 1024):
        idx = offset + i
        mask = idx < (C * H * W)
        
        # Load input values
        input_vals = tl.load(input_ptr + idx, mask=mask, other=-float('inf'))
        
        # Softmax along the flattened dimension (original dim=2)
        max_val = tl.max(input_vals, 0)
        exp_vals = tl.exp(input_vals - max_val)
        sum_exp = tl.sum(exp_vals, 0)
        softmax_vals = exp_vals / sum_exp
        
        # Store output
        tl.store(output_ptr + idx, softmax_vals, mask=mask)

@torch.fx.wrap
def optimized_softmax_reshape(softmax_input):
    N, C, HW = softmax_input.shape
    H, W = 64, 64
    output = torch.empty((N, C, H, W), device=softmax_input.device, dtype=softmax_input.dtype)
    
    grid = lambda meta: (N, )
    softmax_reshape_kernel[grid](
        softmax_input,
        output,
        N, C, H, W
    )
    return output

def replacement_func():
    return optimized_softmax_reshape