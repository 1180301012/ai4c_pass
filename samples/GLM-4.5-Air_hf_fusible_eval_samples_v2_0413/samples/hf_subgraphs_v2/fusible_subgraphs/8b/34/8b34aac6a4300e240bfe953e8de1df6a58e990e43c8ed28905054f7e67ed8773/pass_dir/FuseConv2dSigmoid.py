import torch
import triton
import triton.language as tl

# Pattern for Branch 1: interpolate + sigmoid + multiply fusion
def pattern(input_tensor, multiplier_tensor):
    interpolated = torch.nn.functional.interpolate(input_tensor, (64, 64), None, 'bilinear', False)
    sigmoid_result = torch.sigmoid(interpolated)
    multiply_result = multiplier_tensor * sigmoid_result
    return interpolated, sigmoid_result, multiply_result

def replacement_args(input_tensor, multiplier_tensor):
    return (input_tensor, multiplier_tensor)

@triton.jit
def fused_interpolate_sigmoid_multiply_kernel(
    input_ptr,      # [B, C, H_in, W_in] - input tensor (typically 16x16)
    multiplier_ptr,  # [B, C, H_out, W_out] - multiplier tensor (64x64)
    output_ptr,      # [B, C, H_out, W_out] - final result after multiply
    B, C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Each program processes one output spatial location
    out_y = tl.program_id(2) * BLOCK_SIZE
    out_x_offset = tl.arange(0, BLOCK_SIZE)
    
    for out_x in out_x_offset:
        if out_y >= H_out or out_x >= W_out:
            continue
            
        # Calculate source coordinates for bilinear interpolation
        src_y = (out_y * H_in) // H_out
        src_x = (out_x * W_in) // W_out
        
        # Ensure coordinates are within bounds
        src_y = tl.maximum(0, tl.minimum(src_y, H_in - 1)).to(tl.int32)
        src_x = tl.maximum(0, tl.minimum(src_x, W_in - 1)).to(tl.int32)
        
        # Load input value
        input_base = batch_id * C * H_in * W_in + channel_id * H_in * W_in
        input_offset = input_base + src_y * W_in + src_x
        input_val = tl.load(input_ptr + input_offset)
        
        # Apply sigmoid
        sigmoid_val = 1.0 / (1.0 + tl.exp(-input_val))
        
        # Load multiplier value at output location
        multiplier_base = batch_id * C * H_out * W_out + channel_id * H_out * W_out
        multiplier_offset = multiplier_base + out_y * W_out + out_x
        multiplier_val = tl.load(multiplier_ptr + multiplier_offset)
        
        # Apply multiplication
        result = sigmoid_val * multiplier_val
        
        # Store result
        output_offset = multiplier_base + out_y * W_out + out_x
        tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_interpolate_sigmoid_multiply(input_tensor, multiplier_tensor):
    B, C, H_in, W_in = input_tensor.shape
    H_out, W_out = 64, 64
    
    # Validate multiplier tensor shape
    expected_multiplier_shape = (B, C, H_out, W_out)
    if multiplier_tensor.shape != expected_multiplier_shape:
        # This should match the actual computation in the model
        raise AssertionError(f"Expected multiplier shape {expected_multiplier_shape}, got {multiplier_tensor.shape}")
    
    # Create output tensor
    output = torch.empty((B, C, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with optimized block size
    BLOCK_SIZE = 16  # Process 16 columns per program
    grid = (B, C, (H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    fused_interpolate_sigmoid_multiply_kernel[grid](
        input_tensor, multiplier_tensor, output,
        B, C, H_in, W_in, H_out, W_out,
        BLOCK_SIZE
    )
    
    # Return all intermediate tensors to match original computation
    interpolated = torch.nn.functional.interpolate(input_tensor, (64, 64), None, 'bilinear', False)
    sigmoid_result = torch.sigmoid(interpolated)
    return interpolated, sigmoid_result, output

def replacement_func():
    return fused_interpolate_sigmoid_multiply