import torch
import triton
import triton.language as tl

def pattern(in_3, tmp_1, tmp_0, in_2):
    # Match the complete computation chain
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

def replacement_args(in_3, tmp_1, tmp_0, in_2):
    return (in_3, tmp_1, tmp_0, in_2)



@triton.jit
def fused_conv_sigmoid_mul_hardtanh_kernel(
    # Convolution inputs
    input_ptr,      # in_3: [B, C_in, 1, 1]
    weight_ptr,     # tmp_1: [C_out, C_in, 1, 1] 
    bias_ptr,       # tmp_0: [C_out]
    # Feature input and output
    feature_ptr,    # in_2: [B, C_out, H, W]
    output_ptr,     # output: [B, C_out, H, W]
    # Tensor dimensions
    batch_size: tl.constexpr,
    c_out: tl.constexpr,
    c_in: tl.constexpr,
    h: tl.constexpr,
    w: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Program IDs: batch, spatial position
    batch = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Compute spatial coordinates
    h_idx = spatial_idx // w
    w_idx = spatial_idx % w
    
    # Load feature value at current spatial position
    feature_offset = batch * c_out * h * w + h_idx * w + w_idx
    feature_val = tl.load(feature_ptr + feature_offset)
    
    # Compute conv1x1 + sigmoid for current batch position
    conv_result = 0.0
    
    # Load bias
    bias = tl.load(bias_ptr)
    
    # Optimized convolution computation with better memory locality
    # For small C_in (19), compute with direct access patterns
    input_offset = batch * c_in
    for k in range(c_in):
        # Load input and weight values with coalesced memory access
        input_val = tl.load(input_ptr + input_offset + k)
        weight_val = tl.load(weight_ptr + k * c_out)
        conv_result += weight_val * input_val
    
    # Add bias and apply sigmoid
    sigmoid_result = 1.0 / (1.0 + tl.exp(-(conv_result + bias)))
    
    # Apply element-wise multiplication and hardtanh
    final_result = feature_val * sigmoid_result
    final_result = tl.maximum(tl.minimum(final_result, 6.0), 0.0)
    
    # Store result
    output_offset = batch * c_out * h * w + h_idx * w + w_idx
    tl.store(output_ptr + output_offset, final_result)

@torch.fx.wrap
def fused_forward(in_3, tmp_1, tmp_0, in_2):
    # Get tensor dimensions
    batch_size, c_out, h, w = in_2.shape
    c_in = in_3.shape[1]
    
    # Create output tensor
    output = torch.empty_like(in_2)
    
    # Configure kernel parameters for optimal performance
    # Use 64 for better GPU utilization with our workload
    BLOCK_SIZE_HW = 64  # Block size for spatial dimensions
    
    # Compute grid dimensions with better GPU utilization
    # Round up to ensure full GPU occupancy
    spatial_grid_size = (h * w + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    batch_grid_size = batch_size
    
    # Launch kernel
    fused_conv_sigmoid_mul_hardtanh_kernel[(batch_grid_size, spatial_grid_size)](
        input_ptr=in_3,
        weight_ptr=tmp_1,
        bias_ptr=tmp_0,
        feature_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        c_out=c_out,
        c_in=c_in,
        h=h,
        w=w,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return output

def replacement_func():
    return fused_forward