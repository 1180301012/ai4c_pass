import torch
import triton
import triton.language as tl

@triton.jit
def fused_multiply_add_pad_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    a_batch, a_channels, a_height, a_width,
    b_channels, b_height, b_width,
    c_height, c_width,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate output position
    total_elements = a_height * b_width  # After transpose and multiplication
    
    if pid >= total_elements:
        return
    
    # Calculate indices
    out_0 = pid // (a_height * b_width)
    out_h = (pid % (a_height * b_width)) // b_width
    out_w = (pid % (a_height * b_width)) % b_width
    
    # Load a and b elements with bounds checking
    # a has shape (1, 8, a_height, a_width) after transpose
    # b has shape (1, 8, b_height, b_width) - matches final dimensions
    a_idx = out_0 * (a_channels * a_height * a_width) + out_h * a_width + out_w
    a_val = tl.load(a_ptr + a_idx, mask=(out_h < a_height and out_w < a_width), other=0.0)
    
    # For multiplication, ensure we're accessing valid positions
    b_idx = out_0 * (b_channels * b_height * b_width) + out_h * b_width + out_w  
    b_val = tl.load(b_ptr + b_idx, mask=(out_h < b_height and out_w < b_width), other=0.0)
    
    # Step 1: Multiply
    multiply_result = a_val * b_val
    
    # Step 2: Apply padding - add row with zeros at the top
    # padding pattern: (0, 0, 1, 0, 0, 0) means pad 1 row at top, 0 at bottom, 0 columns
    if out_h == 0:
        # This is the padded row at the top - fill with zeros from multiply operation
        temp_result = 0.0
    else:
        # Shift result down by 1 due to padding
        temp_result = multiply_result
    
    # Step 3: Multiply by scalar (from c_ptr)
    c_idx = out_0 * (c_height * c_width) + out_h * c_width + out_w
    c_val = tl.load(c_ptr + c_idx, mask=(out_h < c_height and out_w < c_width), other=0.0)
    scalar_result = temp_result * scalar
    
    # Step 4: Add (element-wise addition with c tensor)
    add_result = scalar_result + c_val
    
    # Store final result
    tl.store(output_ptr + pid, add_result)

def pattern(in_6, tmp_5, in_4, scalar):
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scalar * in_4  
    tmp_9 = tmp_8 + tmp_7
    return tmp_9, tmp_7, tmp_8  # Return all observables

def replacement_args(in_6, tmp_5, in_4, scalar):
    return (in_6, tmp_5, in_4, scalar)

@torch.fx.wrap
def fused_multiply_add_pad(a_tensor, b_tensor, c_tensor, scalar):
    # Input shapes after previous operations
    a_batch, a_channels, a_height, a_width = a_tensor.shape
    b_batch, b_channels, b_height, b_width = b_tensor.shape  
    c_batch, c_channels, c_height, c_width = c_tensor.shape
    
    # Expected output shape after operations
    output_batch = 1
    output_channels = c_channels
    final_height = c_height
    final_width = c_width
    
    # Check shapes are compatible
    assert a_batch == b_batch == c_batch == output_batch
    assert a_channels == b_channels  # Both should be 8
    assert a_height == c_height + 1  # a has one extra row before padding
    assert a_width == c_width        # Width should match
    assert b_height == c_height     
    assert b_width == c_width
    assert c_channels == output_channels
    
    output = torch.empty((output_batch, output_channels, final_height, final_width),
                        dtype=a_tensor.dtype, device=a_tensor.device)
    
    # Triton launch configuration
    BLOCK_SIZE = 1024
    total_elements = output_batch * output_channels * final_height * final_width
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_multiply_add_pad_kernel[grid](
        a_tensor,
        b_tensor,
        c_tensor,
        output,
        a_batch, a_channels, a_height, a_width,
        b_channels, b_height, b_width,
        final_height, final_width,
        scalar,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_multiply_add_pad