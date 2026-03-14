import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match the computation pattern: slicing + reshape + permute + contiguous
    # This pattern operates on the result of in_2 + in_3
    input_tensor = in_2  # After optimization, this will be the result of the addition
    tmp_3 = input_tensor[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_4 = tmp_3.reshape(1, 12, 12, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_2,)  # Only the tensor that gets sliced and reshaped

@triton.jit
def layout_fusion_kernel(
    input_ptr,
    output_ptr,
    input_batch,
    input_seq_len,
    input_features,
    output_batch,
    output_features,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the output shape: [1, 512, 12, 12]
    # Input is [1, 144, 512], output is [1, 512, 12, 12]
    
    # Each program handles a column in the output features dimension
    feat_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)
    
    # For batch dimension 0
    batch = 0
    
    # Calculate input position: the input is [1, 144, 512]
    # We want to reshape [1, 144, 512] -> [1, 12, 12, 512] -> [1, 512, 12, 12]
    # So position in input: [batch, row*12 + col, feat_idx]
    input_pos = (batch, row_idx * 12 + col_idx, feat_idx)
    input_offset = input_pos[0] * (input_seq_len * input_features) + input_pos[1] * input_features + input_pos[2]
    
    # Calculate output position: [batch, feat_idx, row_idx, col_idx]
    output_pos = (batch, feat_idx, row_idx, col_idx)
    output_offset = output_pos[0] * (output_features * height * width) + \
                   output_pos[1] * (height * width) + \
                   output_pos[2] * width + output_pos[3]
    
    # Load from input and store to output
    mask = (row_idx < 12) & (col_idx < 12) & (feat_idx < output_features)
    input_val = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def fused_layout_operations(input_tensor_input):
    input_shape = input_tensor_input.shape
    input_batch, input_seq_len, input_features = input_shape
    
    # Output shape: [1, 512, 12, 12]
    output_batch = 1
    output_features = input_features  # 512
    height, width = 12, 12
    
    # Calculate grid size
    grid_z = output_features  # 512
    grid_y = height          # 12  
    grid_x = width           # 12
    
    # Create output tensor
    output_shape = (output_batch, output_features, height, width)
    output_tensor = torch.empty(output_shape, dtype=input_tensor_input.dtype, device=input_tensor_input.device)
    
    # Launch kernel
    layout_fusion_kernel[(grid_z, grid_y, grid_x)](
        input_ptr=input_tensor_input,
        output_ptr=output_tensor,
        input_batch=input_batch,
        input_seq_len=input_seq_len,
        input_features=input_features,
        output_batch=output_batch,
        output_features=output_features,
        height=height,
        width=width,
        BLOCK_SIZE=1,
    )
    
    return output_tensor

def replacement_func():
    return fused_layout_operations