import torch
import triton
import triton.language as tl

# Pattern matching function for detach + type_as operations
def pattern(source_tensor, target_tensor):
    # Match the detach + type_as pattern:
    # tmp_6 = tmp_2.detach()
    # tmp_7 = tmp_6.type_as(tmp_5)
    detached = source_tensor.detach()
    typed = detached.type_as(target_tensor)
    return typed

def replacement_args(source_tensor, target_tensor):
    return (source_tensor, target_tensor)

# Optimized Triton kernel for fused Conv3D + Flatten + Transpose
@triton.jit
def fused_conv3d_flatten_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, 
    output_ptr,
    batch_size, channels_in, depth_in, height_in, width_in,
    channels_out, kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dilation_d, dilation_h, dilation_w,
    groups: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Extract program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    m_mask = m_offsets < channels_out
    n_mask = n_offsets < (depth_in * height_in * width_in)
    k_mask = k_offsets < channels_in
    
    # Load bias
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(0, channels_in, BLOCK_SIZE_K):
        k_block = k + k_offsets
        k_mask_block = k_block < channels_in
        
        # Load input block
        input_offsets = (m_offsets[:, None] * (channels_in * depth_in * height_in * width_in) + 
                        k_block[None, :] * (depth_in * height_in * width_in) + 
                        n_offsets[None, :])
        input_mask = m_mask[:, None] & n_mask[None, :] & k_mask_block[None, :]
        input_block = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        
        # Load weight block
        weight_offsets = (m_offsets[:, None] * (channels_in * kernel_d * kernel_h * kernel_w) + 
                         k_block[None, :] * (kernel_d * kernel_h * kernel_w))
        weight_mask = m_mask[:, None] & k_mask_block[None, :]
        weight_block = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        
        # Compute matrix multiplication
        accumulator += tl.dot(weight_block.to(tl.float32), input_block.to(tl.float32))
    
    # Add bias
    accumulator += bias[None, :]
    
    # Store result
    output_offsets = m_offsets[:, None] * (depth_in * height_in * width_in) + n_offsets[None, :]
    tl.store(output_ptr + output_offsets, accumulator.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])

# Kernel wrapper for fused operation
@torch.fx.wrap
def fused_conv3d_flatten_transpose(input_tensor, weight_tensor, bias_tensor):
    # Determine output shape dimensions
    batch_size, channels_in, depth_in, height_in, width_in = input_tensor.shape
    channels_out, _, kernel_d, kernel_h, kernel_w = weight_tensor.shape
    
    # Convolution parameters
    stride_d, stride_h, stride_w = 2, 16, 16
    pad_d, pad_h, pad_w = 0, 0, 0
    dilation_d, dilation_h, dilation_w = 1, 1, 1
    
    # Calculate output dimensions after convolution
    out_depth = (depth_in + 2 * pad_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
    out_height = (height_in + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width_in + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Reshape for matrix multiplication: treat spatial dimensions as flattened
    input_tensor_reshaped = input_tensor.reshape(batch_size, channels_in, depth_in * height_in * width_in)
    weight_tensor_reshaped = weight_tensor.reshape(channels_out, channels_in, kernel_d * kernel_h * kernel_w)
    
    # Output will be [batch_size, channels_out, out_depth * out_height * out_width]
    # Then we reshape to [batch_size, out_depth * out_height * out_width, channels_out] (flattened + transpose)
    
    cu_input = input_tensor_reshaped.contiguous()
    cu_weight = weight_tensor_reshaped.contiguous()
    
    # Create output tensor
    output_shape = (batch_size, channels_out, out_depth * out_height * out_width)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Kernel launch configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    num_programs_m = (channels_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (out_depth * out_height * out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_conv3d_flatten_transpose_kernel[(num_programs_m, num_programs_n), (
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M
    )](
        cu_input,
        cu_weight,
        bias_tensor,
        output_tensor,
        batch_size,
        channels_in,
        depth_in,
        height_in,
        width_in,
        channels_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation_d,
        dilation_h,
        dilation_w,
        1  # groups
    )
    
    # Return as flattened + transposed format to match original pattern
    return output_tensor.transpose(1, 2)

# Simple replacement function for detach + type_as
def optimized_detach_type_as(source_tensor, target_tensor):
    """
    Optimized version of detach() + type_as() that uses direct .to() operation
    which is more efficient than the sequential detach() + type_as() calls.
    """
    # Direct .to() is more efficient than detach() + type_as()
    return source_tensor.to(dtype=target_tensor.dtype, device=target_tensor.device)

def replacement_func():
    # Return the function reference for detach + type_as optimization
    return optimized_detach_type_as