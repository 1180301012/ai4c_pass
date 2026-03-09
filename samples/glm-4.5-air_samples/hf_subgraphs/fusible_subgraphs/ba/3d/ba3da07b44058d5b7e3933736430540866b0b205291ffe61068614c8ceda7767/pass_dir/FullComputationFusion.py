import torch
import triton
import triton.language as tl

# Pattern matching function: Reshape operation (for full computation pattern)
def pattern(input_tensor, weight, bias, N_final, C_final):
    # Focus on reshape operation as the key identifiable pattern
    # This represents the structured reshape in the computation graph
    tmp_4 = input_tensor.reshape(N_final, -1, C_final)
    return tmp_4

def replacement_args(input_tensor, weight, bias, N_final, C_final):
    return (input_tensor, weight, bias, N_final, C_final)

@triton.jit
def full_fusion_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N, C_in, H_in, W_in,
    C_out, K_H, K_W,
    N_final, C_final,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output position in final tensor
    total_final_elements = N_final * (H_in * W_in) * C_final
    if pid >= total_final_elements:
        return
    
    # Map final tensor position back to conv output
    # Final tensor: [N_final, H_in*W_in, C_final]
    # After conv + permute: [N_final, H_in, W_in, C_out]
    # Original conv output: [N_final, C_out, H_in, W_in]
    
    n = pid // ((H_in * W_in) * C_final)
    pos_in_HW = (pid % ((H_in * W_in) * C_final)) // C_final
    c = pid % C_final
    
    h = pos_in_HW // W_in
    w = pos_in_HW % W_in
    
    # If this conv output channel doesn't map to final channel, skip
    if c >= C_final:
        return
    
    # Perform convolution operations
    acc = 0.0
    
    # Conv2D operation (1x1 convolution)
    for c_in in range(C_in):
        # Load bias
        bias_val = tl.load(bias_ptr + c, other=0.0)
        
        # Sum over all input channels for this spatial position
        channel_sum = 0.0
        for spatial_idx in range(C_out):
            # Load weight (1x1 conv, so just 1x1 kernel)
            weight_val = tl.load(weight_ptr + (c * C_in + spatial_idx), other=0.0)
            
            # Load input (assuming we need to access original input)
            # Since we don't have full input access, this is simplified
            # In practice, we'd need to pass input_ptr with proper layout
            input_offset = n * C_in * H_in * W_in + spatial_idx * H_in * W_in + h * W_in + w
            input_val = tl.load(input_ptr + input_offset, other=0.0)
            
            channel_sum += input_val * weight_val
        
        acc += channel_sum + bias_val
    
    # Apply sigmoid
    exp_val = tl.exp(-tl.abs(acc))
    sigmoid_val = tl.where(acc >= 0, 
                         1.0 / (1.0 + tl.exp(-acc)),  # Standard sigmoid
                         tl.exp(acc) / (1.0 + tl.exp(acc)))
    
    # Store result
    tl.store(output_ptr + pid, sigmoid_val)

@torch.fx.wrap
def full_computation_fused(input_tensor, weight, bias, N_final, C_final):
    # Get shapes
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, K_H, K_W, _ = weight.shape
    
    # Create output tensor
    final_HW = H_in * W_in
    output = torch.empty((N_final, final_HW, C_final), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 1
    total_elements = N_final * final_HW * C_final
    grid_size = (total_elements + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    
    # Launch kernel
    full_fusion_kernel[grid_size](
        input_tensor,
        weight,
        bias,
        output,
        N, C_in, H_in, W_in,
        C_out, K_H, K_W,
        N_final, C_final,
        BLOCK_SIZE_X, BLOCK_SIZE_Y
    )
    
    return output

def replacement_func():
    return full_computation_fused