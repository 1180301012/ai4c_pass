import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_4, tmp_5

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def fused_relu_batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    relu_out_ptr,
    batch_norm_out_ptr,
    n_elements,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU using Triton
    relu_out = tl.maximum(x, 0.0)
    tl.store(relu_out_ptr + offsets, relu_out, mask=mask)
    
    # Load batch norm parameters (using first element of each parameter since we're doing channel-wise normalization)
    mean = tl.load(running_mean_ptr + 0)
    var = tl.load(running_var_ptr + 0)
    w = tl.load(weight_ptr + 0)
    b = tl.load(bias_ptr + 0)
    
    # Batch norm computation
    inv_std = 1.0 / tl.sqrt(var + eps)
    norm = (relu_out - mean) * inv_std
    batch_norm_out = norm * w + b
    
    # Store results
    tl.store(batch_norm_out_ptr + offsets, batch_norm_out, mask=mask)

@torch.fx.wrap
def fused_relu_batch_norm(in_4, in_0, in_1, in_3, in_2):
    N = in_4.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    relu_out = torch.empty_like(in_4)
    batch_norm_out = torch.empty_like(in_4)
    
    fused_relu_batch_norm_kernel[(num_programs,)](
        x_ptr=in_4,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        relu_out_ptr=relu_out,
        batch_norm_out_ptr=batch_norm_out,
        n_elements=N,
        momentum=0.1,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return relu_out, batch_norm_out

def replacement_func():
    return fused_relu_batch_norm