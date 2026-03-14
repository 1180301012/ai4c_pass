import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Original computation pattern with softmax along last dimension
    in_1 += in_0
    tmp_0 = in_1
    tmp_1 = tmp_0.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_1 = None
    tmp_3 = tmp_2.type_as(tmp_0)
    tmp_2 = tmp_0 = None
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    tmp_3 = None
    return (tmp_4,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_softmax_dropout_kernel_2d(
    input_ptr,
    out_ptr,
    batch_size,
    num_heads,
    height,
    width,
    HEAD_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    # Get program IDs for different dimensions
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    row_id = tl.program_id(2)
    
    # Calculate pointer offsets
    batch_offset = batch_id * num_heads * height * width
    head_offset = head_id * height * width
    row_offset = row_id * width
    
    base_ptr = input_ptr + batch_offset + head_offset + row_offset
    
    # Load the row of data for softmax
    offsets = tl.arange(0, WIDTH)
    mask = offsets < width
    row_data = tl.load(base_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softmax along the width dimension (last dimension)
    max_val = tl.max(row_data, mask=mask)
    stable_row_data = row_data - max_val
    exp_row_data = tl.exp(stable_row_data)
    sum_exp = tl.sum(exp_row_data, mask=mask)
    softmax_row_data = exp_row_data / sum_exp
    
    # Apply dropout
    random_seed = (batch_id * 1000 + head_id * 100 + row_id) * 1000
    mask_keep = tl.random(random_seed) > 0.1
    dropout_result = softmax_row_data * mask_keep * (1.0 / 0.9)
    
    # Store the result
    tl.store(base_ptr + offsets, dropout_result, mask=mask)

@torch.fx.wrap
def optimized_2d_softmax_dropout(input_tensor):
    # Handle 4D tensor: [batch, heads, height, width]
    if input_tensor.dim() != 4:
        raise ValueError("Input tensor must be 4D")
    
    batch_size, num_heads, height, width = input_tensor.shape
    
    # Adjust grid configuration for 2D softmax
    HEAD_SIZE = 128  # Number of heads per block
    WIDTH = 128      # Width per block
    
    grid = (
        batch_size,
        (num_heads + HEAD_SIZE - 1) // HEAD_SIZE,
        (height + WIDTH - 1) // WIDTH
    )
    
    out = torch.empty_like(input_tensor, dtype=input_tensor.dtype)
    
    optimized_softmax_dropout_kernel_2d[grid](
        input_ptr=input_tensor,
        out_ptr=out,
        batch_size=batch_size,
        num_heads=num_heads,
        height=height,
        width=width,
        HEAD_SIZE=HEAD_SIZE,
        WIDTH=WIDTH,
    )
    
    return out

def replacement_func():
    return optimized_2d_softmax_dropout