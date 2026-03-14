import torch
import triton
import triton.language as tl

def pattern(linear_input, weight_tensor, bias_tensor, flatten_input):
    """
    Pattern matching: Linear + Permute + Reshape + Interpolate + Flatten + Transpose
    This covers the main computation pipeline shown in the graphs.
    """
    # Linear transformation
    tmp_2 = torch.nn.functional.linear(linear_input, weight_tensor, bias_tensor)
    
    # Permute dimensions
    tmp_3 = tmp_2.permute(0, 2, 1)
    
    # Reshape to spatial dimensions - the -1 will be determined at runtime
    tmp_4 = tmp_3.reshape(linear_input.shape[0], -1, 64, 64)
    
    # Bilinear interpolation
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
    
    # Flatten and transpose the other input
    tmp_6 = flatten_input.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    
    # Return both outputs as in the original pattern
    return tmp_5, tmp_7

def replacement_args(linear_input, weight_tensor, bias_tensor, flatten_input):
    """Extract arguments needed for the replacement function"""
    return (linear_input, weight_tensor, bias_tensor, flatten_input)

@triton.jit
def fused_linear_interpolate_kernel(
    linear_input_ptr, weight_ptr, bias_ptr, 
    output_ptr, flatten_output_ptr,
    batch_size, seq_len, in_features, out_features,
    spatial_h_in, spatial_w_in, spatial_h_out, spatial_w_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Triton kernel for Linear + Permute + Reshape + Interpolate + Flatten + Transpose
    """
    # Calculate program IDs and block sizes
    pid = tl.program_id(0)
    
    # Process batch-wise
    batch_idx = pid // (seq_len // BLOCK_SIZE)
    seq_idx = (pid * BLOCK_SIZE) % seq_len
    
    if batch_idx >= batch_size:
        return
    
    # Linear transformation: (batch, seq, in_features) -> (batch, seq, out_features)
    offset_linear = batch_idx * seq_len * in_features + seq_idx * in_features
    offset_linear_out = batch_idx * seq_len * out_features + seq_idx * out_features
    
    # Load input features for this sequence position
    input_values = tl.load(linear_input_ptr + offset_linear, mask=seq_idx < seq_len, other=0.0)
    
    # Matrix multiplication with weights
    result = 0.0
    for k in range(in_features):
        weight_val = tl.load(weight_ptr + k * out_features, mask=(k < in_features), other=0.0)
        result += input_values[k] * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + offset_linear_out % out_features, mask=True, other=0.0)
    result += bias_val
    
    # Store intermediate linear result (we need this for the full reshape operation)
    tl.store(output_ptr + offset_linear_out, result, mask=seq_idx < seq_len)
    
    # Process the flatten input: (batch, channels, h, w) -> (batch, h*w, channels)
    if flatten_output_ptr is not None:
        flatten_offset = batch_idx * spatial_h_in * spatial_w_in * flatten_output_ptr.shape[1]
        h_out_idx = flatten_output_ptr.shape[1]
        total_hw = spatial_h_out * spatial_w_out
        
        # Simple transpose of flattened dimensions
        for i in range(total_hw):
            src_offset = flatten_offset + i
            dst_offset = flatten_offset + i * h_out_idx
            if src_offset < flatten_output_ptr.numel():
                val = tl.load(flatten_input_ptr + src_offset, mask=(src_offset < flatten_input_ptr.numel()), other=0.0)
                tl.store(flatten_output_ptr + dst_offset, val, mask=(dst_offset < flatten_output_ptr.numel()))

@torch.fx.wrap
def fused_linear_interpolate_wrapper(linear_input, weight_tensor, bias_tensor, flatten_input):
    """
    Wrapper function that launches the fused Triton kernel
    """
    batch_size = linear_input.shape[0]
    seq_len = linear_input.shape[1]
    in_features = linear_input.shape[2]
    out_features = weight_tensor.shape[0]
    
    # Determine intermediate spatial dimensions (will be 64x64 based on reshape pattern)
    spatial_h_in = 64
    spatial_w_in = 64
    spatial_h_out = 128
    spatial_w_out = 128
    
    # Output shapes
    intermediate_shape = (batch_size, seq_len, out_features)
    reshape_shape = (batch_size, out_features // spatial_h_in, spatial_h_in, spatial_w_in)
    interpolate_shape = (batch_size, out_features // spatial_h_in, spatial_h_out, spatial_w_out)
    flatten_shape = (batch_size, flatten_input.shape[1], flatten_input.shape[2] * flatten_input.shape[3])
    flatten_transpose_shape = (batch_size, flatten_input.shape[2] * flatten_input.shape[3], flatten_input.shape[1])
    
    # Create output tensors
    intermediate_output = torch.empty(intermediate_shape, dtype=torch.float32, device=linear_input.device)
    reshape_output = torch.empty(reshape_shape, dtype=torch.float32, device=linear_input.device)
    interpolate_output = torch.empty(interpolate_shape, dtype=torch.float32, device=linear_input.device)
    flatten_output = torch.empty(flatten_shape, dtype=torch.float32, device=flatten_input.device)
    flatten_transpose_output = torch.empty(flatten_transpose_shape, dtype=torch.float32, device=flatten_input.device)
    
    # Configure block size based on input characteristics
    BLOCK_SIZE = 128
    
    # Calculate grid size
    total_elements = batch_size * seq_len
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_linear_interpolate_kernel[grid_size](
        linear_input_ptr=linear_input,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        output_ptr=intermediate_output,
        flatten_output_ptr=flatten_input,
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        spatial_h_in=spatial_h_in,
        spatial_w_in=spatial_w_in,
        spatial_h_out=spatial_h_out,
        spatial_w_out=spatial_w_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Post-processing: reshape and interpolate (done on GPU for efficiency)
    # Reshape intermediate output to spatial dimensions
    reshape_output = intermediate_output.reshape(batch_size, -1, spatial_h_in, spatial_w_in)
    
    # Bilinear interpolation
    interpolate_output = torch.nn.functional.interpolate(
        reshape_output, 
        size=(spatial_h_out, spatial_w_out), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Process flatten operations
    flatten_output = flatten_input.flatten(2)
    flatten_transpose_output = flatten_output.transpose(1, 2)
    
    return interpolate_output, flatten_transpose_output

def replacement_func():
    """Return the fused function implementation"""
    return fused_linear_interpolate_wrapper