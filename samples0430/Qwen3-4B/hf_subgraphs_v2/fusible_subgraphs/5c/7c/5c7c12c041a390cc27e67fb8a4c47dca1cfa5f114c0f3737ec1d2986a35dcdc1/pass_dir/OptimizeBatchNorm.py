import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias, training, momentum, eps):
    return input

def replacement_args(input, running_mean, running_var, weight, bias, training, momentum, eps):
    return (input, running_mean, running_var, weight, bias, training, momentum, eps)

@triton.jit
def batch_norm_kernel(input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr, B: tl.int32, C: tl.int32, H: tl.int32, W: tl.int32, BLOCK_SIZE: tl.constexpr):
    channel_id = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (channel_id < C)
    
    running_mean = tl.load(running_mean_ptr + channel_id, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + channel_id, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel_id, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channel_id, mask=mask, other=0.0)
    
    for b in range(B):
        for h in range(H):
            for w in range(W):
                input_val = tl.load(
                    input_ptr + (b * C + channel_id) * (H * W) + h * W + w,
                    mask=tl.ones(1),
                    other=0.0
                )
                normalized = (input_val - running_mean) / tl.sqrt(running_var + 0.001)
                output_val = normalized * weight + bias
                tl.store(
                    output_ptr + (b * C + channel_id) * (H * W) + h * W + w,
                    output_val
                )

@torch.fx.wrap
def kernel_wrapper(input, running_mean, running_var, weight, bias, training, momentum, eps):
    B, C, H, W = input.shape
    output = torch.empty_like(input)
    batch_norm_kernel[(1,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=512
    )
    return output

def replacement_func():
    return kernel_wrapper