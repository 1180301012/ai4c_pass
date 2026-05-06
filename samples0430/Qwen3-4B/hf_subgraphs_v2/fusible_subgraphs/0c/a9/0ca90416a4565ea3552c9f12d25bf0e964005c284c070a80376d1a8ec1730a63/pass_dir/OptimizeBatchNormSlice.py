import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2, in_5, start):
    tmp_4 = in_5[(slice(None, None, None), slice(start, None, None), slice(None, None, None), slice(None, None, None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)

def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5, start):
    return (in_4, in_0, in_1, in_3, in_2, in_5, start)

def batch_norm_triton_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    tensor_ptr,
    start,
    output_ptr,
    sliced_ptr,
    N: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate offsets based on thread ID
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE
    
    # Create mask to handle out-of-bounds
    mask = tl.arange(BLOCK_SIZE) < N
    
    # Load input and stats
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Compute normalized values
    x_norm = (input - running_mean) / tl.sqrt(running_var + 0.001)
    output = x_norm * weight + bias
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)
    
    # Slice the tensor for the channel selection
    tensor_slice = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
    tl.store(sliced_ptr + offsets, tensor_slice, mask=mask)

def batch_norm_triton_wrapper(
    in_4,
    in_0,
    in_1,
    in_3,
    in_2,
    in_5,
    start
):
    # Get tensor shapes
    N, C, H, W = in_4.shape
    
    # Allocate output tensors
    output = torch.empty((N, C, H, W), dtype=in_4.dtype, device=in_4.device)
    sliced = torch.empty_like(in_5)
    
    # Launch kernel with proper grid
    batch_norm_triton_kernel[tl.cdiv(N, 128)](
        input_ptr=in_4,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        tensor_ptr=in_5,
        start=start,
        output_ptr=output,
        sliced_ptr=sliced,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=128
    )
    
    return (output, sliced)

def replacement_func():
    return batch_norm_triton_wrapper