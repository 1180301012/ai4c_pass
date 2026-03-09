import torch
import triton
import triton.language as tl

def pattern(inputs_embeds):
    tmp_10 = inputs_embeds.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    return tmp_16

def replacement_args(inputs_embeds):
    return (inputs_embeds,)

@triton.jit
def fused_layer_norm_kernel(input_ptr, output_ptr, n_elements, 
                           channel_mean, channel_inv_std, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Get channel indices for each element
    seq_len = n_elements // 2048  # Assuming 2048 channels based on weight_meta
    channel_indices = offsets % 2048
    
    # Normalize using pre-computed per-channel statistics
    channel_mean_val = tl.load(channel_mean + channel_indices, mask=mask, other=0.0)
    channel_inv_std_val = tl.load(channel_inv_std + channel_indices, mask=mask, other=0.0)
    
    # Normalize and store
    x_normalized = (x - channel_mean_val) * channel_inv_std_val
    tl.store(output_ptr + offsets, x_normalized.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_layer_norm(inputs_embeds):
    # Convert to float32 first as in original
    input_float = inputs_embeds.to(torch.float32)
    
    # Compute per-channel RMS statistics (matching original computation)
    # tmp_10.pow(2).mean(-1, keepdim=True) + 1e-06, then rsqrt
    squared = input_float ** 2
    channel_mean_sq = torch.mean(squared, dim=-1, keepdim=True)  # Mean of squares per channel
    channel_rms = torch.rsqrt(channel_mean_sq + 1e-06)  # This is 1/RMS per channel
    
    # Flatten channel statistics for kernel access
    n_elements = input_float.numel()
    n_channels = input_float.shape[-1]
    channel_mean_flat = torch.ones(n_channels, dtype=torch.float32, device=input_float.device)
    channel_inv_std_flat = channel_rms.flatten()  # This is our normalization factor
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(inputs_embeds, dtype=torch.bfloat16)
    
    fused_layer_norm_kernel[(num_programs,)](
        input_ptr=input_float,
        output_ptr=output,
        n_elements=n_elements,
        channel_mean=channel_mean_flat,
        channel_inv_std=channel_inv_std_flat,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_layer_norm