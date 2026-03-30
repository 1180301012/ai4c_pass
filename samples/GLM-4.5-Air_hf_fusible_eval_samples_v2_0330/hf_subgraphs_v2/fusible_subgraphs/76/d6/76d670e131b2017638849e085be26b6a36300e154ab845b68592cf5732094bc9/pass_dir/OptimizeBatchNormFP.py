import torch
import triton
import triton.language as tl

def pattern(arg1, arg2, arg3, arg4, arg5):
    # Create a chain of operations that matches the structure
    tmp1 = arg1 - arg2
    tmp2 = tmp1 / arg3
    tmp3 = tmp2 * arg4
    tmp4 = tmp3 + arg5
    return tmp4

def replacement_args(in_7, in_0, in_1, in_3, in_2):
    return (in_7, in_0, in_1, in_3, in_2)

@triton.jit
def batchnorm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(N * C, BLOCK_SIZE)
    
    if pid >= num_pid:
        return
    
    # Calculate global offsets
    block_start = pid * BLOCK_SIZE
    total_elements = N * C
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load parameters (broadcast across spatial dimensions)
    C_pid = tl.cdiv(offsets, N)
    mean = tl.load(mean_ptr + C_pid, mask=C_pid < C, other=0.0)
    var = tl.load(var_ptr + C_pid, mask=C_pid < C, other=0.0)
    weight = tl.load(weight_ptr + C_pid, mask=C_pid < C, other=1.0)
    bias = tl.load(bias_ptr + C_pid, mask=C_pid < C, other=0.0)
    
    # Batch normalization formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    out = (x - mean) / tl.sqrt(var + eps) * weight + bias
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_batchnorm(in_7, in_0, in_1, in_3, in_2):
    N, C = in_7.shape
    eps = 1e-05
    
    BLOCK_SIZE = 1024
    
    num_pid = tl.cdiv(N * C, BLOCK_SIZE)
    grid = (num_pid,)
    
    out = torch.empty_like(in_7)
    
    batchnorm_kernel[grid](
        in_7,
        in_0,
        in_1,
        in_3,
        in_2,
        out,
        N,
        C,
        eps,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return triton_batchnorm