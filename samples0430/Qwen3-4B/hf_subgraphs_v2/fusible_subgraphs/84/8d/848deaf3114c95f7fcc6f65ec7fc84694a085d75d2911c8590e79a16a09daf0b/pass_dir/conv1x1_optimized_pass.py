import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp + in_2
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def efficient_conv1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    O: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    o_start = pid * BLOCK_SIZE
    o_end = min(o_start + BLOCK_SIZE, O)
    
    for o in tl.arange(o_start, o_end):
        bias = tl.load(bias_ptr + o)
        
        for h in tl.arange(0, H):
            for w in tl.arange(0, W):
                accum = 0.0
                for i in tl.arange(0, C_in):
                    input_val = tl.load(input_ptr + (h * W + w) * C_in + i)
                    weight_val = tl.load(weight_ptr + (o * C_in) + i)
                    accum += input_val * weight_val
                tl.store(output_ptr + (h * W + w) * O + o, accum + bias)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    batch_size = in_3.shape[0]
    C_in = in_3.shape[1]
    H = in_3.shape[2]
    W = in_3.shape[3]
    O = in_0.shape[0]
    
    out = torch.empty_like(in_2)
    
    BLOCK_SIZE = 64
    num_programs = (O + BLOCK_SIZE - 1) // BLOCK_SIZE
    efficient_conv1x1_kernel[(num_programs,)](
        input_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=out,
        N=batch_size,
        C_in=C_in,
        H=H,
        W=W,
        O=O,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
def replacement_func():
    return kernel_wrapper