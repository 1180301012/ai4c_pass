import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    # Match the entire computation graph
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    batch_size = conv2d.shape[0]
    tmp_3 = conv2d.view(batch_size, 1, -1)
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr,
    out_ptr, 
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program processes one spatial location across all batch elements
    spatial_idx = pid
    if spatial_idx >= height * width:
        return
    
    base_offset = pid * batch_size * out_channels
    
    for b in range(tl.num_programs(1)):
        batch_idx = tl.program_id(1)
        if batch_idx >= batch_size:
            break
            
        # Compute memory offset for this batch element and spatial location
        offset = base_offset + batch_idx * out_channels
        
        # Load bias (same for all spatial locations)
        bias = tl.load(bias_ptr + 0)
        
        # Perform 1x1 convolution - essentially weighted sum of input channels
        result = bias
        for c in range(in_channels):
            # Load input and weight for this channel
            input_val = tl.load(input_ptr + batch_idx * in_channels * height * width + c * height * width + spatial_idx)
            weight_val = tl.load(weight_ptr + c)
            result += input_val * weight_val
        
        # Store result
        tl.store(out_ptr + offset, result)

@triton.jit
def fused_arithmetic_kernel_fp32(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and convert to fp32 for computation
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Fused operations: sigmoid(x) - 0.25 * pi
    x_exp = tl.exp(-tl.abs(x))
    sigmoid = tl.where(x >= 0, 1.0 / (1.0 + x_exp), x_exp / (1.0 + x_exp))
    result = (sigmoid - 0.25) * 3.141592653589793
    
    # Store output, converting back to original dtype
    tl.store(output_ptr + offsets, result.to(tl.float32), mask=mask)

@triton.jit
def fused_arithmetic_kernel_bf16(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input as float32 for computation
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Fused operations: sigmoid(x) - 0.25 * pi
    x_exp = tl.exp(-tl.abs(x))
    sigmoid = tl.where(x >= 0, 1.0 / (1.0 + x_exp), x_exp / (1.0 + x_exp))
    result = (sigmoid - 0.25) * 3.141592653589793
    
    # Store output as bf16
    tl.store(output_ptr + offsets, result.to(tl.bfloat16), mask=mask)

@triton.jit
def fused_arithmetic_kernel_fp16(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input as float32 for computation
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Fused operations: sigmoid(x) - 0.25 * pi
    x_exp = tl.exp(-tl.abs(x))
    sigmoid = tl.where(x >= 0, 1.0 / (1.0 + x_exp), x_exp / (1.0 + x_exp))
    result = (sigmoid - 0.25) * 3.141592653589793
    
    # Store output as float16
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)

@triton.jit
def full_graph_fusion_kernel(
    input_conv_ptr, weight_conv_ptr, bias_conv_ptr,
    input_ptr_3, input_ptr_4,
    output_ptr,
    batch_size, conv_output_size, input3_size, input4_size, total_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size
    
    # Load conv2d output section
    conv_start = input3_size + input4_size
    conv_section_mask = (offsets >= conv_start) & (offsets < conv_start + conv_output_size)
    
    # Load input_3 section
    input3_mask = offsets < input3_size
    input3_vals = tl.where(input3_mask, 
                          tl.load(input_ptr_3 + offsets, mask=input3_mask, other=0.0),
                          0.0)
    
    # Load input_4 section  
    input4_mask = (offsets >= input3_size) & (offsets < input3_size + input4_size)
    input4_vals = tl.where(input4_mask,
                          tl.load(input_ptr_4 + offsets - input3_size, mask=input4_mask, other=0.0),
                          0.0)
    
    # Load conv2d output section
    conv_vals = tl.where(conv_section_mask,
                        tl.load(input_conv_ptr + offsets - conv_start, mask=conv_section_mask, other=0.0),
                        0.0)
    
    # Concatenate: input_3 + input_4 + conv2d_output
    concatenated = input3_vals + input4_vals + conv_vals
    
    # Apply fused sigmoid arithmetic operations
    x_exp = tl.exp(-tl.abs(concatenated))
    sigmoid = tl.where(concatenated >= 0, 1.0 / (1.0 + x_exp), x_exp / (1.0 + x_exp))
    result = (sigmoid - 0.25) * 3.141592653589793
    
    # Store final result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap  
def full_graph_fusion(in_0, in_1, in_2, in_3, in_4):
    # For now, just use the individual operations but avoid forbidden APIs
    # This pass is simplified to only optimize the arithmetic part
    batch_size, channels, height, width = in_2.shape
    
    # Create empty tensor for conv2d output
    conv_output_shape = (batch_size, 1, height, width)
    conv_output = torch.empty(conv_output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Perform conv2d using PyTorch (this will be optimized separately)
    # The actual optimization should be done in a separate pass for just conv2d
    conv_output = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # View operation
    conv_flat = conv_output.view(batch_size, 1, -1)
    
    # Concatenation would need a separate optimization pass
    concatenated = torch.cat([in_3, in_4, conv_flat], 2)
    
    # Apply fused arithmetic operations (this is the main optimization here)
    n_elements = concatenated.numel()
    
    # Choose optimal block size
    if n_elements < 8192:
        BLOCK_SIZE = 256
    elif n_elements < 65536:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Reuse arithmetic kernel from the specialized pass
    output = torch.empty_like(concatenated)
    
    # Use appropriate kernel based on dtype
    if concatenated.dtype == torch.bfloat16:
        fused_kernel = fused_arithmetic_kernel_bf16
    elif concatenated.dtype == torch.float16:
        fused_kernel = fused_arithmetic_kernel_fp16
    else:
        fused_kernel = fused_arithmetic_kernel_fp32
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_kernel[(num_programs,)](
        input_ptr=concatenated,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return full_graph_fusion