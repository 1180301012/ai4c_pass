import torch
import triton
import triton.language as tl

def pattern(input_4, running_mean, running_var, weight, bias):
    tmp_4 = input_4.reshape(1, 512, 8, 8)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

@triton.jit
def fused_batchnorm_silu_kernel_bfloat16_512(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load batch norm parameters (from CPU memory)
    idx = (offsets // (H * W)) % C
    running_mean = tl.load(running_mean_ptr + idx, mask=idx < C, other=0.0).to(tl.float32)
    running_var = tl.load(running_var_ptr + idx, mask=idx < C, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + idx, mask=idx < C, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + idx, mask=idx < C, other=0.0).to(tl.float32)
    
    # BatchNorm computation
    inv_var = 1.0 / tl.sqrt(running_var + eps)
    x_norm = (x - running_mean) * inv_var * weight + bias
    
    # SiLU activation
    out = x_norm * (1.0 / (1.0 + tl.exp(-x_norm)))
    
    # Store result
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_batchnorm_silu_bfloat16_512(x, running_mean, running_var, weight, bias):
    N, C, H, W = 1, 512, 8, 8
    n_elements = N * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_batchnorm_silu_kernel_bfloat16_512[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        eps=1e-05,
        momentum=0.1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_args(input_4, running_mean, running_var, weight, bias):
    return (input_4, running_mean, running_var, weight, bias)

def replacement_func():
    return fused_batchnorm_silu_bfloat16_512