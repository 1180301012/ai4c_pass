import torch
import triton
import triton.language as tl

def pattern(x):
    # Just match the reshape/view chain:
    # tmp_1 = tmp_0.reshape(8, -1)                      # [8, 256]
    # tmp_2 = tmp_1.view(8, -1, 1, 1)                   # [8, 256, 1, 1] 
    # tmp_3 = tmp_2.view(8, 2, -1, 1, 1)                # [8, 2, 128, 1, 1]
    
    # Fuse reshape + view operations into single direct transform
    tmp1 = x.reshape(8, -1)
    tmp2 = tmp1.view(8, -1, 1, 1)
    final_out = tmp2.view(8, 2, -1, 1, 1)
    return final_out

def replacement_args(x):
    return (x,)

@triton.jit
def fused_softmax_reshape_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    dim1_size,
    dim3_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Compute indices within batch
    offset = batch_idx * dim1_size * dim3_size
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * dim1_size * dim3_size
    
    # Load input data: layout is [batch, 2, 1, 128]
    in_data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Apply softmax along dim=1 (which corresponds to the middle dimension of our loaded data)
    # We need to reshape the data for softmax computation
    # The input shape is [8, 2, 1, 128], we process one batch at a time
    # For softmax along dim=1, we process [2, 1, 128] chunks per batch
    
    # Reshape data for softmax: convert to [2, 1*128] per batch
    flat_size = dim1_size * dim3_size
    in_reshaped = tl.reshape(in_data, (dim1_size, flat_size))
    
    # Compute max for numerical stability
    max_val = tl.max(in_reshaped, axis=1)
    max_val = tl.reshape(max_val, (dim1_size, 1))
    
    # Compute softmax
    exp_in = tl.exp(in_reshaped - max_val)
    sum_exp = tl.sum(exp_in, axis=1, keepdims=True)
    softmax_out = exp_in / sum_exp
    
    # Reshape back to original [2, 1, 128] and flatten for storage
    softmax_reshaped = tl.reshape(softmax_out, (dim1_size, dim3_size))
    
    # Store result in final shape [8, 2, 128, 1, 1] flattened
    tl.store(out_ptr + offsets, softmax_reshaped, mask=mask)

@triton.jit
def fused_computation_kernel(
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    batch_size,
    channels_in_1,
    spatial_in_1,
    channels_in_0,
    height_in_0,
    width_in_0,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position in the final output
    program_id = tl.program_id(0)
    
    # Final output shape after sum along dim=1: [batch_size, channels_in_1-1, height_in_0, width_in_0]
    # But looking at the pattern, sum along dim=1 reduces the channel dimension
    output_size = batch_size * channels_in_1 * height_in_0 * width_in_0
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    if mask[0]:  # Only proceed if we have valid output positions
        # For simplicity, process one output element at a time
        output_idx = offsets[0]
        
        # Decompose indices
        batch_idx = output_idx // (channels_in_1 * height_in_0 * width_in_0)
        rem = output_idx % (channels_in_1 * height_in_0 * width_in_0)
        ch_idx = rem // (height_in_0 * width_in_0)
        rem = rem % (height_in_0 * width_in_0)
        h_idx = rem // width_in_0
        w_idx = rem % width_in_0
        
        # Initialize accumulator for this output position
        total_sum = 0.0
        
        # Sum over the dimension we're reducing (dim=1 with size 2)
        for reduce_dim in range(2):  # This is the dimension we sum over
            # Load multiplier value for this (batch, reduce_dim, ch_idx, 1, 1)
            mult_offset = (batch_idx * 2 + reduce_dim) * channels_in_1 * spatial_in_1 + ch_idx * spatial_in_1
            multiplier_val = tl.load(in_1_ptr + mult_offset, mask=mult_offset < batch_size * 2 * channels_in_1 * spatial_in_1, other=0.0)
            
            # Load operand value for this (batch, reduce_dim, ch_idx, h_idx, w_idx)
            # operand has shape [batch, 2, 128, height, width]
            operand_offset = (batch_idx * channels_in_0 + reduce_dim * channels_in_1 + ch_idx) * height_in_0 * width_in_0 + h_idx * width_in_0 + w_idx
            operand_val = tl.load(in_0_ptr + operand_offset, mask=operand_offset < batch_size * channels_in_0 * height_in_0 * width_in_0, other=0.0)
            
            # Multiply and accumulate
            total_sum += multiplier_val * operand_val
        
        # Store the result
        tl.store(out_ptr + output_idx, total_sum, mask=offsets < output_size)

@triton.jit
def fused_reshape_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    input_dim2,
    input_dim3,
    output_dim1,
    output_dim2,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element of the input tensor
    program_id = tl.program_id(0)
    
    # Calculate input and output tensor sizes
    input_size = batch_size * 2 * input_dim2 * input_dim3  # [batch, 2, 1, 128]
    output_size = batch_size * output_dim1 * output_dim2 * 1 * 1  # [batch, 2, 128, 1, 1]
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    # Load input value
    input_val = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate output offset using the same indexing but simplified
    # The reshape pattern: [batch, 2, 1, 128] -> [batch, 2, 128, 1, 1]
    # We map the input layout directly to output layout
    # For input [batch, 2, 1, 128]:
    # batch_idx = offset // (2 * 1 * 128)
    # ch_idx = (offset % (2 * 1 * 128)) // (1 * 128)  
    # h_idx = ((offset % (2 * 1 * 128)) % (1 * 128)) // 128
    # w_idx = ((offset % (2 * 1 * 128)) % (1 * 128)) % 128 = 0 (since input_dim2=1)
    
    # Simplified indexing since input_dim2=1 (third dimension is 1)
    batch_idx = offsets // (2 * 1 * input_dim3)
    rem = offsets % (2 * 1 * input_dim3)
    ch_idx = rem // (1 * input_dim3)  # This will be 0 or 1
    h_idx = rem % input_dim3  # This maps to 0..127
    
    # Map to output: [batch, ch_idx, h_idx, 1, 1] -> flattened: batch*2*128 + ch_idx*128 + h_idx
    output_offset = batch_idx * output_dim1 * output_dim2 + ch_idx * output_dim2 + h_idx
    
    # Store to output with proper masking (use input mask since output size matches)
    tl.store(out_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def fused_reshape(x):
    # Input tensor should have shape that matches the pattern
    # Input: [batch, 2, 1, 128] (softmax output)
    
    # Get input tensor properties
    batch_size = x.shape[0]
    input_dim2 = x.shape[2]  # 1 (third dimension)
    input_dim3 = x.shape[3]  # 128 (spatial dimension)
    
    # Output tensor shape: [batch, 2, 128, 1, 1]
    output_dim1 = 2
    output_dim2 = input_dim3  # 128
    
    # Calculate total input elements (should match output)
    input_size = batch_size * 2 * input_dim2 * input_dim3
    # Output tensor shape: [batch, 2, 128, 1, 1] - needed for broadcasting with in_0
    out = torch.empty([batch_size, output_dim1, output_dim2, 1, 1], dtype=x.dtype, device=x.device)
    
    # Flatten tensors for kernel
    x_flat = x.reshape(-1)
    out_flat = out.reshape(-1)
    
    # Determine kernel launch configuration
    BLOCK_SIZE = 1024
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_reshape_kernel[(num_programs,)](
        in_ptr=x_flat,
        out_ptr=out_flat,
        batch_size=batch_size,
        input_dim2=input_dim2,
        input_dim3=input_dim3,
        output_dim1=output_dim1,
        output_dim2=output_dim2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_reshape