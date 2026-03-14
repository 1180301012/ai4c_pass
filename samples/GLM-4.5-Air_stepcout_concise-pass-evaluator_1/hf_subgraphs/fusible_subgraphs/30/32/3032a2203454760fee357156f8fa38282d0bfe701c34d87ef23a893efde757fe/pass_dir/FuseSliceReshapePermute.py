import torch
import triton
import triton.language as tl
import math

def pattern(tmp_2):
    # Match the slice -> reshape -> permute sequence on existing tensor
    tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_4 = tmp_3.reshape(1, 12, 12, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    return tmp_5

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def fused_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    output_batch: tl.constexpr,
    output_hidden: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate indices for the input tensor
    # We want to access tmp_3 which is tmp_2[:, 1:, :] -> shape [1, 144, 512]
    # Our output is [1, 512, 12, 12]
    
    # For each output element, calculate corresponding input index
    # output[0][h][i][j] corresponds to input[0][i*j + 1 + (h)][j_offset]
    
    # Flattened output indices
    flat_output = output_flat_idx = offsets
    
    output_idx = tl.arange(0, BLOCK_SIZE)
    
    # Convert flat output index to multi-dimensional coordinates
    # output shape: [output_batch, output_hidden, output_height, output_width]
    b = output_idx // (output_hidden * output_height * output_width)
    remainder = output_idx % (output_hidden * output_height * output_width)
    h = remainder // (output_height * output_width)
    remainder = remainder % (output_height * output_width)
    i = remainder // output_width
    j = remainder % output_width
    
    # Map to input coordinates
    # tmp_3 shape: [batch_size, seq_len-1, hidden_size] = [1, 144, 512]
    # Our slice comes from axis 1, position 1 onwards
    input_b = b
    input_i = i * output_width + j + 1  # +1 to skip first element along axis 1
    input_j = h
    
    mask_b = (b < output_batch) & (input_i < seq_len)
    mask = mask_b & (offsets < output_batch * output_hidden * output_height * output_width)
    
    # Load input data
    input_ptr_offset = input_b * seq_len * hidden_size + input_i * hidden_size + input_j
    input_data = tl.load(input_ptr + input_ptr_offset, mask=mask, other=0.0)
    
    # Store output data
    output_ptr_offset = flat_output
    tl.store(output_ptr + output_ptr_offset, input_data, mask=mask)

@triton.jit
def fused_slice_reshape_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    output_batch: tl.constexpr,
    output_hidden: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert flat output index to multi-dimensional coordinates
    # output shape: [output_batch, output_hidden, output_height, output_width]
    output_idx = tl.arange(0, BLOCK_SIZE)
    b = output_idx // (output_hidden * output_height * output_width)
    remainder = output_idx % (output_hidden * output_height * output_width)
    h = remainder // (output_height * output_width)
    remainder = remainder % (output_height * output_width)
    i = remainder // output_width
    j = remainder % output_width
    
    # Map to input coordinates
    # tmp_2 shape: [batch_size, seq_len, hidden_size] = [1, 145, 512]
    # tmp_3 shape: [batch_size, seq_len-1, hidden_size] = [1, 144, 512]
    # We're accessing tmp_3[:, i*output_width + 1:, h] - so need to add 1 to skip first element
    input_b = b
    input_i = i * output_width + j + 1  # +1 to skip first element along axis 1
    input_j = h
    
    mask_b = (b < output_batch) & (input_i < seq_len)
    mask = mask_b & (offsets < output_batch * output_hidden * output_height * output_width)
    
    # Load input data from input tensor
    input_ptr_offset = input_b * seq_len * hidden_size + input_i * hidden_size + input_j
    input_data = tl.load(input_ptr + input_ptr_offset, mask=mask, other=0.0)
    
    # Store output data (no addition needed, just data movement)
    output_ptr_offset = offsets
    tl.store(output_ptr + output_ptr_offset, input_data, mask=mask)

@torch.fx.wrap
def fused_slice_reshape_permute(tmp_2):
    input_shape = tmp_2.shape
    batch_size, seq_len, hidden_size = input_shape
    
    # Compute output after slice->reshape->permute sequence
    # tmp_2: [batch_size, seq_len, hidden_size] = [1, 145, 512]
    # tmp_3: [batch_size, seq_len-1, hidden_size] = [1, 144, 512] 
    # tmp_4: [1, 12, 12, 512] after reshape
    # tmp_5: [1, 512, 12, 12] after permute
    output_shape = (1, hidden_size, 12, 12)
    output_size = 1 * hidden_size * 12 * 12
    
    output = torch.empty(output_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_slice_reshape_kernel[(num_programs,)](
        input_ptr=tmp_2,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        output_batch=1,
        output_hidden=hidden_size,
        output_height=12,
        output_width=12,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_slice_reshape_permute