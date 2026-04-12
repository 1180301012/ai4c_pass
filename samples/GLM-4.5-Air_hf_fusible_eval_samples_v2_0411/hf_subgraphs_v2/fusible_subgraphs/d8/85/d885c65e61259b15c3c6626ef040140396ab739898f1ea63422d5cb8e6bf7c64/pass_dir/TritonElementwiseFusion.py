import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(conv2d_out, gate_input):
    """
    Match the pattern: sigmoid(conv2d_out) * gate_input -> hardtanh to [0,6]
    """
    tmp_3 = conv2d_out.sigmoid()
    tmp_4 = gate_input * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function  
def replacement_args(conv2d_out, gate_input):
    """
    Extract arguments needed for the replacement kernel
    """
    return (conv2d_out, gate_input)

# Optimized kernel with fused operations
@triton.jit
def fused_sigmoid_multiply_kernel(
    conv2d_ptr,
    gate_ptr,
    out_ptr,
    conv2d_stride,
    gate_stride,
    out_stride,
    batch_size, conv_channels, gate_channels, height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    High-performance kernel that fuses sigmoid multiplication and hardtanh
    Handles the broadcast: [batch, conv_channels, 1, 1] * [batch, gate_channels, height, width]
    """
    program_id = tl.program_id(0)
    
    # Handle batch and channel dimensions
    b = program_id // conv_channels
    c = program_id % conv_channels
    
    # Load conv2d value (scalar per batch and conv channel due to 1x1 spatial)
    conv_idx = b * conv2d_stride[0] + c * conv2d_stride[1]
    conv_val = tl.load(conv2d_ptr + conv_idx)
    
    # Apply sigmoid
    exp_neg_conv = tl.exp(-tl.abs(conv_val))
    sigmoid_val = tl.where(conv_val > 0,
                          1.0 / (1.0 + exp_neg_conv),
                          exp_neg_conv / (1.0 + exp_neg_conv))
    
    # Process spatial region for this conv channel
    for h in tl.range(0, height, BLOCK_SIZE_N):
        for w in tl.range(0, width, BLOCK_SIZE_N):
            # Compute spatial indices with bounds checking
            h_idx = h + tl.arange(0, BLOCK_SIZE_N)
            w_idx = w + tl.arange(0, BLOCK_SIZE_N)
            
            h_mask = h_idx < height
            w_mask = w_idx < width
            
            # Create mask for block
            mask = h_mask[:, None] & w_mask[None, :]
            
            # Load gate tensor values for this spatial block
            gate_ptr_block = gate_ptr + (b * gate_stride[0] + 
                                        c * gate_stride[1] + 
                                        h_idx[:, None] * gate_stride[2] + 
                                        w_idx[None, :] * gate_stride[3])
            gate_vals = tl.load(gate_ptr_block, mask=mask, other=0.0)
            
            # Apply fused operation: sigmoid_val * gate -> clamp to [0, 6]
            result = sigmoid_val * gate_vals
            
            # Apply hardtanh manually using tl.where
            clamped_result = tl.where(result < 0, 0.0,
                                    tl.where(result > 6.0, 6.0, result))
            
            # Store result
            out_ptr_block = out_ptr + (b * out_stride[0] + 
                                     c * out_stride[1] + 
                                     h_idx[:, None] * out_stride[2] + 
                                     w_idx[None, :] * out_stride[3])
            tl.store(out_ptr_block, clamped_result, mask=mask)

# Simplified kernel for direct mapping case
@triton.jit
def direct_fused_kernel(
    input_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Direct fused kernel for matching tensor shapes
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    gate_val = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    
    # Apply sigmoid
    exp_neg_input = tl.exp(-tl.abs(input_val))
    sigmoid_result = tl.where(input_val > 0,
                             1.0 / (1.0 + exp_neg_input),
                             exp_neg_input / (1.0 + exp_neg_input))
    
    # Multiply and clamp
    result = sigmoid_result * gate_val
    final_result = tl.where(result < 0, 0.0,
                           tl.where(result > 6.0, 6.0, result))
    
    # Store
    tl.store(out_ptr + offsets, final_result, mask=mask)

# Main kernel wrapper
@torch.fx.wrap
def triton_fused_elementwise(conv2d_out, gate_input):
    """
    Optimized fused operation using Triton kernels
    """
    
    # Get tensor shapes
    conv_shape = conv2d_out.shape  # Should be [B, C_conv, 1, 1] typically
    gate_shape = gate_input.shape  # Should be [B, C_gate, H, W] typically
    
    # Create output tensor
    out = torch.empty_like(gate_input)
    
    # Handle different tensor size scenarios
    conv_elements = conv2d_out.numel()
    gate_elements = gate_input.numel()
    
    # Case 1: Both tensors have same total size (direct 1:1 mapping)
    if conv_elements == gate_elements:
        BLOCK_SIZE = 256
        num_programs = (gate_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        direct_fused_kernel[(num_programs,)](
            conv2d_out,
            gate_input, 
            out,
            gate_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Case 2: Broadcast scenario [B, C, 1, 1] * [B, C, H, W] -> [B, C, H, W]
        B, C_conv, H_conv, W_conv = conv_shape
        B, C_gate, H_gate, W_gate = gate_shape
        
        # Verify broadcast pattern
        if (B == B and C_conv == C_gate and H_conv == 1 and W_conv == 1):
            BLOCK_SIZE_N = 8  # Small block size for spatial dimensions
            grid_size = B * C_conv  # One program per batch and conv channel
            
            fused_sigmoid_multiply_kernel[(grid_size,)](
                conv2d_out,
                gate_input,
                out,
                conv2d_out.stride(),
                gate_input.stride(),
                out.stride(),
                B, C_conv, C_gate, H_gate, W_gate,
                BLOCK_SIZE_M=1,  # Single channel per program
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        else:
            # Fallback: reshape and use direct kernel
            conv_flat = conv2d_out.reshape(-1)
            gate_flat = gate_input.reshape(-1)
            out_flat = out.reshape(-1)
            
            BLOCK_SIZE = 256
            num_programs = (gate_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            direct_fused_kernel[(num_programs,)](
                conv_flat,
                gate_flat,
                out_flat,
                gate_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    
    return out

# Replacement function
def replacement_func():
    """
    Return the optimized kernel function
    """
    return triton_fused_elementwise