import torch
import triton
import triton.language as tl

def pattern(conv_in, conv_weight, running_mean, running_var, weight, bias):
    conv_out = torch.conv2d(conv_in, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    batch_norm_out = torch.nn.functional.batch_norm(
        conv_out,
        running_mean,
        running_var,
        weight,
        bias,
        False,
        0.1,
        1e-05
    )
    return batch_norm_out
def replacement_args(conv_in, conv_weight, running_mean, running_var, weight, bias):
    return (conv_in, conv_weight, running_mean, running_var, weight, bias)

@triton.jit
def conv_bn_kernel(
    conv_in_ptr,
    conv_weight_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    I,
    H,
    W,
    O,
    kh,
    kw,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Calculate our block start in the input
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load inputs with mask
    valid = offsets < N
    conv_in = tl.load(conv_in_ptr + block_start + offsets, mask=valid)
    conv_weight = tl.load(conv_weight_ptr + offsets, mask=valid)
    running_mean = tl.load(running_mean_ptr + offsets, mask=valid)
    running_var = tl.load(running_var_ptr + offsets, mask=valid)
    weight = tl.load(weight_ptr + offsets, mask=valid)
    bias = tl.load(bias_ptr + offsets, mask=valid)
    
    # Compute batch normalization
    inv_std = 1.0 / tl.sqrt(running_var + 1e-5)
    out = (conv_in * conv_weight - running_mean) * inv_std * weight + bias
    
    # Store result
    tl.store(out_ptr + block_start + offsets, out, mask=valid)

@torch.fx.wrap
def conv_bn_kernel_wrapper(
    conv_in,
    conv_weight,
    running_mean,
    running_var,
    weight,
    bias,
):
    # Get shapes
    N, I, H, W = conv_in.shape
    O, _, kh, kw = conv_weight.shape
    
    # Calculate number of programs
    BLOCK_SIZE_VAL = 128
    num_programs = (N + BLOCK_SIZE_VAL - 1) // BLOCK_SIZE_VAL
    
    # Allocate output
    out = torch.empty_like(conv_in)
    
    # Launch kernel
    conv_bn_kernel[(num_programs,)](
        conv_in_ptr=conv_in,
        conv_weight_ptr=conv_weight,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N,
        I=I,
        H=H,
        W=W,
        O=O,
        kh=kh,
        kw=kw,
        BLOCK_SIZE=128,
    )
    
    return out

def replacement_func():
    return conv_bn_kernel_wrapper