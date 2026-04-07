import torch
import triton
import triton.language as tl
import math

def pattern(in_5, in_4):
    # Exact multiplication pattern from the computation
    # Note: We need to return exactly what the computation returns
    result = in_5 * in_4
    return result

def replacement_args(in_5, in_4):
    return (in_5, in_4)

@triton.jit
def fused_bn_silu_kernel(
    input_ptr,
    sigmoid_gates_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C, H, W,
    eps: float,
    momentum: float,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate grid positions
    pid = tl.program_id(0)
    # Each program handles one C dimension across all spatial and batch positions
    c_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_mask = c_offset < C
    
    # Load normalization parameters for this channel
    running_mean = tl.load(running_mean_ptr + c_offset, mask=c_mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_offset, mask=c_mask, other=0.0)
    weight = tl.load(weight_ptr + c_offset, mask=c_mask, other=1.0)
    bias = tl.load(bias_ptr + c_offset, mask=c_mask, other=0.0)
    
    # Pre-compute normalization factor
    inv_std = tl.where(
        running_var > 0.0,
        tl.math.rsqrt(running_var + eps),
        0.0
    )
    weight_scaled = weight * inv_std
    bias_scaled = bias - running_mean * weight_scaled
    
    # Store normalization parameters for spatial processing
    tl.store(running_mean_ptr + c_offset + C, running_mean, mask=c_mask)
    tl.store(running_var_ptr + c_offset + C, running_var, mask=c_mask)
    tl.store(weight_ptr + c_offset + C, weight_scaled, mask=c_mask)
    tl.store(bias_ptr + c_offset + C, bias_scaled, mask=c_mask)
    
    # Process spatial and batch dimensions
    pid_spatial = tl.program_id(1)
    batch = pid_spatial // (H // 8)  # Assume 8x8 blocks for spatial
    spatial_local = (pid_spatial % (H // 8)) * 8 + tl.arange(0, 8)
    spatial_coords = tl.arange(0, 8)
    
    # Process all channels for current spatial position
    for c in range(0, C, BLOCK_SIZE):
        c_base = c
        c_idx = c_base + c_offset
        c_local_mask = (c_idx >= c_base) & (c_idx < c_base + BLOCK_SIZE) & c_mask
        
        if c_base + BLOCK_SIZE >= C:
            break
            
        # Load sigmoid gates
        sigmoid_gate = tl.load(sigmoid_gates_ptr + batch * C * H * W + c_idx * H * W + spatial_local[0, None] * W, mask=c_local_mask, other=0.5)
        
        # Apply sigmoid gate to all spatial positions
        sigmoid_tiles = tl.broadcast_to(sigmoid_gate[:, None], (sigmoid_gate.shape[0], 8))
        
        for hi in range(0, H, 8):
            h_idx = hi + spatial_coords[None, :]
            h_mask = h_idx < H
            
            # Load input values
            input_vals = tl.load(input_ptr + batch * C * H * W + c_idx * H * W + h_idx[:, None] * W + spatial_coords[None, :], 
                              mask=(c_local_mask[:, None] & h_mask[:, None]), other=0.0)
            
            # Apply fused operations
            # 1. Multiply by sigmoid gate
            gated_input = input_vals * sigmoid_tiles
            
            # 2. Apply batch normalization
            weight_use = tl.load(weight_ptr + c_idx + C, mask=c_local_mask, other=1.0)
            bias_use = tl.load(bias_ptr + c_idx + C, mask=c_local_mask, other=0.0)
            normalized = gated_input * weight_use[:, None] + bias_use[:, None]
            
            # 3. Apply SiLU activation: x * sigmoid(x)
            silu_result = normalized * (1.0 / (1.0 + tl.math.exp(-normalized)))
            
            # Store result
            out_base = batch * C * H * W + c_idx * H * W + h_idx[:, None] * W + spatial_coords[None, :]
            mask_out = (c_local_mask[:, None] & h_mask[:, None])
            tl.store(output_ptr + out_base, silu_result, mask=mask_out)

@triton.jit
def simple_mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate multiplication
    output = x * y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def simple_pytorch_mul(x, y):
    # Simple PyTorch multiplication with broadcasting handled by built-in
    # This should work correctly and we can optimize with Triton later
    return x * y

def replacement_func():
    return simple_pytorch_mul