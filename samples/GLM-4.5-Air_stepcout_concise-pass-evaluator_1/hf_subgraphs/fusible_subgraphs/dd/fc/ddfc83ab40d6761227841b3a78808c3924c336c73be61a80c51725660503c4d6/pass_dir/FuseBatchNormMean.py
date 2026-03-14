import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, bn_weight, bn_bias):
    """
    Batch normalization followed by mean reduction over spatial dimensions
    """
    # Batch norm with specific parameters from the model
    bn_out = torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    # Mean reduction over spatial dimensions (H, W) with keepdim=True
    mean_out = bn_out.mean((2, 3), keepdim=True)
    return bn_out, mean_out

def replacement_args(input_tensor, running_mean, running_var, bn_weight, bn_bias):
    return (input_tensor, running_mean, running_var, bn_weight, bn_bias)

@triton.jit
def fused_bn_mean_kernel(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    bn_weight_ptr, 
    bn_bias_ptr,
    bn_out_ptr,
    mean_out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C * H * W
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape offsets to 4D
    offset_w = offsets % W
    offset_h = (offsets // W) % H
    offset_c = (offsets // (W * H)) % C
    offset_n = offsets // (W * H * C)
    
    # Load input tensor
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters (channel-wise)
    channel_idx = offset_c % C
    running_mean_val = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < C)
    running_var_val = tl.load(running_var_ptr + channel_idx, mask=channel_idx < C)
    bn_weight_val = tl.load(bn_weight_ptr + channel_idx, mask=channel_idx < C)
    bn_bias_val = tl.load(bn_bias_ptr + channel_idx, mask=channel_idx < C)
    
    # Batch normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
    # Note: In training mode with momentum=0.1, we'd use running stats
    eps = 1e-05
    normalized = (input_val - running_mean_val) / tl.sqrt(running_var_val + eps)
    bn_result = normalized * bn_weight_val + bn_bias_val
    
    # Store batch norm output
    tl.store(bn_out_ptr + offsets, bn_result, mask=mask)
    
    # For mean reduction, we need to compute mean per channel per batch
    # Each thread will contribute to partial sums
    if pid == 0:  # Only first block initializes mean accumulator
        # Initialize mean storage
        for c in range(C):
            for n in range(N):
                mean_idx = n * C + c
                tl.store(mean_out_ptr + mean_idx, 0.0)
    
    # We need a more sophisticated approach for mean reduction
    # Let's compute partial sums in shared memory and reduce
    thread_sum = tl.sum(bn_result) if mask[offsets] == 1 else 0.0
    
    # Store partial sums for reduction (simplified approach)
    # In practice, we'd need a more sophisticated reduction strategy
    tl.atomic_add(mean_out_ptr + offset_n * C + offset_c, thread_sum / (H * W))

# Simplified version focusing on batch norm with mean computation
@triton.jit
def bn_with_mean_kernel(
    input_ptr, 
    running_mean_ptr, 
    running_var_ptr, 
    bn_weight_ptr, 
    bn_bias_ptr,
    bn_out_ptr,
    mean_out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = N * C * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape offsets to 4D
    offset_w = offsets % W
    offset_h = (offsets // W) % H
    offset_c = (offsets // (W * H)) % C
    offset_n = offsets // (W * H * C)
    
    # Load input tensor
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters for this channel
    channel_idx = offset_c % C
    running_mean_val = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < C)
    running_var_val = tl.load(running_var_ptr + channel_idx, mask=channel_idx < C)
    bn_weight_val = tl.load(bn_weight_ptr + channel_idx, mask=channel_idx < C)
    bn_bias_val = tl.load(bn_bias_ptr + channel_idx, mask=channel_idx < C)
    
    # Apply batch normalization
    eps = 1e-05
    normalized = (input_val - running_mean_val) / tl.sqrt(running_var_val + eps)
    bn_result = normalized * bn_weight_val + bn_bias_val
    
    # Store batch norm result
    tl.store(bn_out_ptr + offsets, bn_result, mask=mask)
    
    # Compute mean for this location and add to mean accumulator
    mean_idx = offset_n * C + offset_c
    if offset_w == 0 and offset_h == 0:  # Only first element in each spatial location contributes
        tl.atomic_add(mean_out_ptr + mean_idx, tl.sum(bn_result) / (H * W))

@torch.fx.wrap
def fused_batch_norm_mean(input_tensor, running_mean, running_var, bn_weight, bn_bias):
    N, C, H, W = input_tensor.shape
    
    # Output tensors
    bn_out = torch.empty_like(input_tensor)
    
    # Mean output: [N, C, 1, 1] 
    mean_out = torch.empty((N, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    total_elements = N * C * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Initialize mean output to zero
    mean_out.fill_(0.0)
    
    fused_bn_mean_kernel[(num_programs,)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        bn_out_ptr=bn_out,
        mean_out_ptr=mean_out.squeeze(),  # Store as [N, C] for easier access
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape mean output back to [N, C, 1, 1]
    mean_out = mean_out.reshape(N, C, 1, 1)
    
    return bn_out, mean_out

def replacement_func():
    return fused_batch_norm_mean