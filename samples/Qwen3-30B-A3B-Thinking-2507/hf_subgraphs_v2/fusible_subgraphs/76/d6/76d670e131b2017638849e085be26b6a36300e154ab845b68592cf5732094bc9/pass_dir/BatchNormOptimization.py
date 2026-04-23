import torch
import triton
import triton.language as tl

def pattern(in_7, in_0, in_1, in_3, in_2):
    return torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)

def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

@triton.jit
def batchnorm_element_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // features
    j = pid % features
    
    if i >= batch_size or j >= features:
        return
        
    x = tl.load(input_ptr + i * features + j)
    m = tl.load(mean_ptr + j)
    v = tl.load(var_ptr + j)
    w = tl.load(weight_ptr + j)
    b = tl.load(bias_ptr + j)
    
    normalized = (x - m) / tl.sqrt(v + eps)
    out = normalized * w + b
    tl.store(output_ptr + i * features + j, out)

@torch.fx.wrap
def batchnorm_wrapper(input, mean, var, weight, bias, eps=1e-05):
    B = input.shape[0]
    F = input.shape[1]
    output = torch.empty_like(input)
    
    grid = ((B * F + 127) // 128, )
    batchnorm_element_kernel[grid](
        input_ptr=input,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=B,
        features=F,
        eps=eps,
        BLOCK_SIZE=128
    )
    return output

def replacement_func():
    return batchnorm_wrapper