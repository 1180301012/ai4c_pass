import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern to match the EXACT computation sequence from the model:
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    Returns (tmp_8, tmp_7) to match original output
    """
    # Capture the parameters as local variables (mirroring original)
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    
    # Step 1: Element-wise addition (exactly as in original)
    tmp_4 = in_5 + in_4
    
    # Step 2: Mean reduction over spatial dimensions (exactly as in original)
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_4 = None  # Clean up as in original
    
    # Step 3: Two no-op dropouts (exactly as in original)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_5 = None  # Clean up as in original
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_6 = None  # Clean up as in original
    
    # Step 4: Batch normalization (exactly as in original with positional args)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_0 = tmp_1 = tmp_3 = tmp_2 = None  # Clean up as in original
    
    # Return exactly as in original
    return (tmp_8, tmp_7)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract all arguments needed for the replacement"""
    # Map original arguments: in_0=bn_mean, in_1=bn_var, in_2=bn_bias, in_3=bn_weight, in_4, in_5=input tensors
    return (in_4, in_5, in_0, in_1, in_3, in_2)

@triton.jit
def fused_add_mean_bnorm_kernel(
    in4_ptr,
    in5_ptr, 
    bn_mean_ptr,
    bn_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_out_ptr,
    dropout_out_ptr,
    batch_size,
    n_channels, 
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Element-wise addition
    2. Mean reduction over spatial dimensions
    3. Batch normalization
    
    Note: Dropouts with p=0.0 are removed as no-ops
    """
    
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # Handle per-channel computation
    output_channels = n_channels
    output_channel_groups = (output_channels + num_programs - 1) // num_programs
    start_channel = program_id * output_channel_groups
    end_channel = min((program_id + 1) * output_channel_groups, output_channels)
    
    for channel_idx in range(start_channel, end_channel):
        
        # Load batch norm parameters for this channel
        bn_mean_val = tl.load(bn_mean_ptr + channel_idx)
        bn_var_val = tl.load(bn_var_ptr + channel_idx) 
        bn_weight_val = tl.load(bn_weight_ptr + channel_idx)
        bn_bias_val = tl.load(bn_bias_ptr + channel_idx)
        
        # Compute mean over spatial and batch dimensions
        spatial_elements = height * width
        batch_spatial_elements = batch_size * spatial_elements
        
        current_sum = 0.0
        valid_elements = 0
        
        batch_idx = 0
        while batch_idx < batch_size:
            spatial_idx = 0
            while spatial_idx < spatial_elements:
                
                # Calculate linear index for current channel, batch, spatial position
                linear_idx = (batch_idx * n_channels + channel_idx) * spatial_elements + spatial_idx
                
                # Load input tensors
                val4 = tl.load(in4_ptr + linear_idx, other=0.0)
                val5 = tl.load(in5_ptr + linear_idx, other=0.0)
                
                # Element-wise addition and accumulate for mean
                add_val = val4 + val5
                current_sum += add_val
                valid_elements += 1
                
                spatial_idx += BLOCK_SIZE
            
            batch_idx += 1
        
        # Compute mean (handle empty case)
        mean_val = current_sum / valid_elements if valid_elements > 0 else 0.0
        
        # Apply batch normalization formula: (x - mean) / sqrt(var + eps) * weight + bias
        denom = tl.sqrt(bn_var_val + 1e-05)
        bn_val = (mean_val - bn_mean_val) / denom * bn_weight_val + bn_bias_val
        
        # Store results
        tl.store(bn_out_ptr + channel_idx, bn_val)
        tl.store(dropout_out_ptr + channel_idx, mean_val)

@torch.fx.wrap 
def fused_add_mean_bnorm(in_4, in_5, bn_mean, bn_var, bn_weight, bn_bias):
    """
    Fused function that combines addition, mean reduction, and batch normalization
    using optimized Triton kernel, while preserving required outputs
    """
    batch_size, n_channels, height, width = in_4.shape
    
    # Output tensors - need to match original return format (batch_norm result, dropout result)
    bn_result = torch.empty(n_channels, dtype=in_4.dtype, device=in_4.device)
    dropout_result = torch.empty(n_channels, dtype=in_4.dtype, device=in_4.device) 
    
    # Launch fused kernel
    grid = (triton.cdiv(n_channels, 1024),)  # Use 1024 programs per channel group
    fused_add_mean_bnorm_kernel[grid](
        in4_ptr=in_4,
        in5_ptr=in_5,
        bn_mean_ptr=bn_mean,
        bn_var_ptr=bn_var, 
        bn_weight_ptr=bn_weight,
        bn_bias_ptr=bn_bias,
        bn_out_ptr=bn_result,
        dropout_out_ptr=dropout_result,
        batch_size=batch_size,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=1024
    )
    
    # Return tuple matching original format
    return (bn_result, dropout_result)

def replacement_func():
    """Return the fused function"""
    return fused_add_mean_bnorm