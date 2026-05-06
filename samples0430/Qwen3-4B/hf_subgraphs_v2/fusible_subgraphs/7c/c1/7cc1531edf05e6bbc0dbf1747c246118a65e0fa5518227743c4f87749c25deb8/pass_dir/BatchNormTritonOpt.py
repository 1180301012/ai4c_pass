import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05):
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def replacement_args(input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05):
    return (input, running_mean, running_var, weight, bias, training, momentum, eps)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global index
    block_start = tl.program_id(0) * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= B * C * H * W:
            break
        
        # Calculate channel and spatial indices
        channel = idx % C
        spatial_index = idx // C
        y = spatial_index % H
        x = spatial_index // H
        
        # Load values
        input_val = tl.load(input_ptr + idx)
        mean_val = tl.load(running_mean_ptr + channel)
        var_val = tl.load(running_var_ptr + channel)
        weight_val = tl.load(weight_ptr + channel)
        bias_val = tl.load(bias_ptr + channel)
        
        # Compute batch norm
        normalized = (input_val - mean_val) / tl.sqrt(var_val + eps)
        out_val = normalized * weight_val + bias_val
        
        # Store result
        tl.store(out_ptr + idx, out_val)

@torch.fx.wrap
def batch_norm_wrapper(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    B = input.shape[0]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    out = torch.empty_like(input)
    
    # Launch kernel with 128-wide blocks
    grid = ((B * C * H * W + 128 - 1) // 128,)
    
    batch_norm_kernel[grid](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        eps=eps,
        BLOCK_SIZE=128,
    )
    return out

def replacement_func():
    return batch_norm_wrapper