import torch
import triton
import triton.language as tl

# Pattern matching function that matches layer_norm + flatten + transpose + view + permute
def pattern(tmp_10, tmp_4, tmp_3):
    # Flatten from dimension 2
    tmp_11 = tmp_10.flatten(2)
    
    # Transpose dimensions 1 and 2
    tmp_12 = tmp_11.transpose(1, 2)
    
    # Layer normalization on the last dimension
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (tmp_12.shape[-2],), tmp_4, tmp_3, 1e-06)
    
    # Reshape to 4D
    tmp_14 = tmp_13.view(tmp_13.shape[0], 56, 56, tmp_13.shape[-1])
    
    # Permute dimensions to final format
    tmp_15 = tmp_14.permute(0, 3, 1, 2)
    return tmp_15

# Argument extraction function
def replacement_args(tmp_10, tmp_4, tmp_3):
    return (tmp_10, tmp_4, tmp_3)

# Optimized kernel for layer normalization with fused reshape operations
@triton.jit
def fused_layer_norm_reshape_kernel(
    x_ptr,           # Input tensor [B, C, H*W] 
    weight_ptr,      # Layer norm weight [H*W]
    bias_ptr,        # Layer norm bias [H*W]  
    out_ptr,         # Output tensor [B, C_out, H, W]
    B, HW, C_out,    # Tensor dimensions
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    # Calculate program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Linear offset for this batch and channel
    base_offset = batch_id * HW * C_out + channel_id * HW
    
    # Process spatial elements
    for hw_offset in tl.range(0, HW, BLOCK_SIZE):
        hw_idx = hw_offset + tl.arange(0, BLOCK_SIZE)
        mask = hw_idx < HW
        
        # Load input data for this batch and channel position
        input_offset = base_offset + hw_idx
        x = tl.load(x_ptr + input_offset, mask=mask, other=0.0)
        
        # Load layer norm parameters
        weight = tl.load(weight_ptr + hw_idx, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + hw_idx, mask=mask, other=0.0)
        
        # Compute mean
        mean = tl.sum(x, axis=0) / HW
        # Compute variance
        x_centered = x - mean
        x2 = x_centered * x_centered
        variance = tl.sum(x2, axis=0) / HW
        
        # Layer normalization
        std_inv = 1.0 / tl.sqrt(variance + EPSILON)
        norm_x = x_centered * std_inv
        result = norm_x * weight + bias
        
        # Store result in output layout (B, C_out, H, W)
        h = hw_idx // 56  # Assuming 56x56 spatial dimensions
        w = hw_idx % 56
        output_offset = batch_id * C_out * 56 * 56 + channel_id * 56 * 56 + h * 56 + w
        tl.store(out_ptr + output_offset, result, mask=mask)

# Kernel wrapper function
@torch.fx.wrap  
def fused_layer_norm_reshape(x, weight, bias):
    B, C, HW = x.shape
    H = W = 56  # Assuming 56x56 spatial resolution from the patterns
    
    # Create output tensor in final format [B, C, H, W]
    out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    
    # Set block size and epsilon
    BLOCK_SIZE = 256
    EPSILON = 1e-06
    
    # Calculate grid dimensions
    grid = (B, C)
    
    # Launch Triton kernel
    fused_layer_norm_reshape_kernel[grid](
        x, weight, bias, out,
        B, HW, C, BLOCK_SIZE, EPSILON
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_layer_norm_reshape