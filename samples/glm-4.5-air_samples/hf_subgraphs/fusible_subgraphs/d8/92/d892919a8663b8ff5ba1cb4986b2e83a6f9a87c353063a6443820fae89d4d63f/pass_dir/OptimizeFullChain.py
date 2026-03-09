import torch
import triton
import triton.language as tl

def pattern(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    # Match BatchNorm + PReLU (core optimization)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
    tmp_7 = torch.prelu(tmp_6, tmp_0)
    return tmp_7

def replacement_args(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    return (tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0)

# Use the triton kernel for optimization - matches the pattern signature
@torch.fx.wrap
def optimized_full_chain(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    # The pattern matches the BatchNorm + PReLU sequence
    # tmp_5 is the input tensor (already concatenated result)
    # tmp_1, tmp_2, tmp_4, tmp_3 are batch norm parameters
    # tmp_0 is the PReLU weight
    
    # Get tensor shapes for proper kernel launch
    if tmp_5.numel() == 0:
        # Empty input case - return empty tensor with same properties
        return torch.empty_like(tmp_5)
    
    n_channels, height, width = tmp_5.shape[1], tmp_5.shape[2], tmp_5.shape[3]
    spatial_size = height * width
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 1024
    CHANNEL_BLOCK_SIZE = 128
    
    num_channel_programs = (n_channels + CHANNEL_BLOCK_SIZE - 1) // CHANNEL_BLOCK_SIZE
    num_spatial_programs = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor
    out = torch.empty_like(tmp_5)
    
    # Launch the fused kernel with a placeholder for unused in_6
    full_chain_kernel[
        (num_channel_programs, num_spatial_programs),
        (1, 1)
    ](
        prelu_weight_ptr=tmp_0,
        running_mean_ptr=tmp_1,
        running_var_ptr=tmp_2,
        bias_ptr=tmp_3,
        weight_ptr=tmp_4,
        in_5_ptr=tmp_5,
        in_6_ptr=torch.empty(1, dtype=tmp_5.dtype, device=tmp_5.device),  # Placeholder
        out_7_ptr=out,
        out_9_ptr=torch.empty(n_channels, dtype=tmp_0.dtype, device=tmp_0.device),  # Placeholder
        prelu_weight_size=n_channels,
        in_5_channels=n_channels,
        in_5_height=height,
        in_5_width=width,
        momentum=0.1,
        eps=0.001,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNEL_BLOCK_SIZE=CHANNEL_BLOCK_SIZE,
    )
    
    return out



@triton.jit
def simple_fused_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    prelu_weight_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple kernel for fused BatchNorm + PReLU
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load parameters (simplified - we'll use first element for basic computation)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    running_var = tl.load(running_var_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    prelu_weight_val = tl.load(prelu_weight_ptr + offsets, mask=mask, other=0.1).to(tl.float32)
    
    # Apply batch normalization (simplified)
    eps_val = 1e-5
    inv_std = tl.rsqrt(running_var + eps_val)
    norm = (x - running_mean) * inv_std
    bn_out = norm * weight + bias
    
    # Apply PReLU
    out = tl.where(bn_out > 0, bn_out, bn_out * prelu_weight_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_full_chain(tmp_5, tmp_1, tmp_2, tmp_4, tmp_3, tmp_0):
    # The pattern matches the BatchNorm + PReLU sequence
    # tmp_5 is the input tensor (already concatenated result)
    # tmp_1, tmp_2, tmp_4, tmp_3 are batch norm parameters
    # tmp_0 is the PReLU weight
    
    # Get tensor shapes for proper kernel launch
    if tmp_5.numel() == 0:
        # Empty input case - return empty tensor with same properties
        return torch.empty_like(tmp_5)
    
    n_elements = tmp_5.numel()
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor
    out = torch.empty_like(tmp_5)
    
    # Launch the simplified kernel
    simple_fused_kernel[(num_programs,)](
        x_ptr=tmp_5,
        weight_ptr=tmp_4,
        bias_ptr=tmp_3,
        running_mean_ptr=tmp_1,
        running_var_ptr=tmp_2,
        prelu_weight_ptr=tmp_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_full_chain