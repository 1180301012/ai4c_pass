import torch
import triton
import triton.language as tl

@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr,
    input_stride,
    output_stride,
    input_shape,
    output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized reshape kernel using Triton"""
    # Calculate total elements
    input_elements = 1
    for dim in input_shape:
        input_elements *= dim
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_elements
    
    # Load input data (flattened)
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store to output (reshaped)
    tl.store(output_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_reshape_conv(conv_out, target_shape):
    """
    Optimized reshape operation for conv_out tensor
    Converts from [N, D] to [1, -1, H, D_out] format
    """
    input_shape = conv_out.shape
    output_shape = target_shape
    
    # Handle the specific reshape pattern seen in the models
    if len(input_shape) == 2 and len(output_shape) == 4:
        N, D = input_shape
        B, H_out, H_in, D_out = output_shape
        
        # Verify that total elements match
        if N * D != B * H_out * H_in * D_out:
            # Fallback to regular reshape if shapes don't match expected pattern
            return torch.reshape(conv_out, target_shape)
        
        # Use Triton kernel for optimal performance
        total_elements = N * D
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty(output_shape, dtype=conv_out.dtype, device=conv_out.device)
        
        optimized_reshape_kernel[(num_programs,)](
            input_ptr=conv_out,
            output_ptr=output,
            input_stride=conv_out.stride(),
            output_stride=output.stride(),
            input_shape=input_shape,
            output_shape=output_shape,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output
    else:
        # Fallback to regular reshape for other patterns
        return torch.reshape(conv_out, target_shape)

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the reshape operation pattern:
    tmp_7 = torch.reshape(in_1, [1, -1, X, Y])
    
    This matches the entire computation but focuses on optimizing just the reshape part.
    The pattern matching framework needs us to match the entire observable computation.
    """
    # Match the complete computation pattern as in the original model
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = torch.reshape(in_1, [1, -1, 6, 64])  # Target reshape operation for optimization
    
    # Return observable outputs - these are the values used outside the pattern
    return (tmp_6, tmp_7)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for the replacement function"""
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    """Return the optimized reshape function"""
    def optimized_with_reshape(in_0, in_1, in_2, in_3):
        # Fused scaling + addition + optimized attention computation
        fused_scaled = in_0 / 8.0 + in_2
        softmax_output = torch.nn.functional.softmax(fused_scaled, dim=-1)
        # Remove no-op dropout (training=False)
        attention_output = torch.matmul(softmax_output, in_3)
        tmp_6 = attention_output.permute(0, 2, 1, 3).contiguous()
        
        # Optimized reshape operation
        tmp_7 = optimized_reshape_conv(in_1, [1, -1, 6, 64])
        
        return (tmp_6, tmp_7)
    
    return optimized_with_reshape