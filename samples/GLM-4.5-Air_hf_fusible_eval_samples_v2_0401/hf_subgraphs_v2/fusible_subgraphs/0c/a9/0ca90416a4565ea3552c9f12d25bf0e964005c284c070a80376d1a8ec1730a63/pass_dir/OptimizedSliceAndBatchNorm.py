import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, slice_start):
    # Slice operation - extracting channel slice
    sliced_tensor = in_5[(slice(None, None, None), slice(slice_start, None, None), slice(None, None, None), slice(None, None, None))]
    # Batch normalization
    batch_norm_out = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return batch_norm_out, sliced_tensor

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, slice_start):
    return (in_0, in_1, in_2, in_3, in_4, in_5, slice_start)

@triton.jit
def fused_slice_and_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    slice_input_ptr,
    output_ptr,
    slice_output_ptr,
    n_elements,
    slice_start,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load batch norm input
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters
    mean_idx = offsets // (tl.load(input_ptr + tl.program_id(1) * tl.load(running_mean_ptr).shape[0] * 1024 * 1024))
    mean_val = tl.load(running_mean_ptr + (mean_idx % tl.load(running_mean_ptr + tl.program_id(1)).shape[0]), mask=mask, other=0.0)
    var_val = tl.load(running_var_ptr + (mean_idx % tl.load(running_var_ptr + tl.program_id(1)).shape[0]), mask=mask, other=1.0)
    weight_val = tl.load(weight_ptr + (mean_idx % tl.load(weight_ptr + tl.program_id(1)).shape[0]), mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + (mean_idx % tl.load(bias_ptr + tl.program_id(1)).shape[0]), mask=mask, other=0.0)
    
    # Batch normalization computation
    var_inv = 1.0 / tl.sqrt(var_val + eps)
    batch_norm_output = (input_val - mean_val) * weight_val * var_inv + bias_val
    
    # Store batch norm output
    tl.store(output_ptr + offsets, batch_norm_output, mask=mask)
    
    # For slice operation - we'll handle this separately as it's a different memory layout
    # The slice operation needs to be handled in the wrapper function

@torch.fx.wrap
def optimized_slice_and_batch_norm(input_tensor, running_mean, running_var, weight, bias, slice_input, slice_start):
    # Get input shapes
    if len(input_tensor.shape) == 4:  # NCHW format for batch norm input
        N, C, H, W = input_tensor.shape
        n_elements = N * C * H * W
    if len(slice_input.shape) == 4:  # NCHW format for slice input
        N_slice, C_slice, H_slice, W_slice = slice_input.shape
    
    # Output tensors
    batch_norm_output = torch.empty_like(input_tensor)
    slice_output = slice_input[(slice(None, None, None), slice(slice_start, None, None), slice(None, None, None), slice(None, None, None))]
    
    # Triton kernel launch configuration for batch norm
    BLOCK_SIZE = 1024
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch batch norm kernel
    fused_slice_and_batch_norm_kernel[(n_blocks, 1, 1)](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        slice_input_ptr=slice_input,
        output_ptr=batch_norm_output,
        slice_output_ptr=slice_output,
        n_elements=n_elements,
        slice_start=slice_start,
        eps=0.001,
        momentum=0.1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return batch_norm_output, slice_output

def replacement_func():
    return optimized_slice_and_batch_norm