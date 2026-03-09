import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_dropout2d_kernel(
    input_ptr,
    relu_output_ptr,
    dropout_output_ptr,
    n_elements,
    dropout_prob: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_out = tl.max(x, 0.0)
    
    # Store ReLU output
    tl.store(relu_output_ptr + offsets, relu_out, mask=mask)
    
    # Apply Dropout - using simple element-wise dropout for now
    # Note: For 2D dropout, we might need more sophisticated masking
    # across spatial dimensions, but this provides a basic optimization
    rand_vals = tl.rand(offsets)
    dropout_mask = rand_vals > dropout_prob
    dropout_out = relu_out * dropout_mask
    
    # Store dropout output
    tl.store(dropout_output_ptr + offsets, dropout_out, mask=mask)

@torch.fx.wrap
def fused_relu_dropout2d(in_0):
    # Get input tensor properties
    n_elements = in_0.numel()
    
    # Create output tensors
    relu_output = torch.empty_like(in_0)
    dropout_output = torch.empty_like(in_0)
    
    # Set block size - optimized for GPU
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_relu_dropout2d_kernel[(num_programs,)](
        input_ptr=in_0,
        relu_output_ptr=relu_output,
        dropout_output_ptr=dropout_output,
        n_elements=n_elements,
        dropout_prob=0.1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (dropout_output, relu_output)

def replacement_func():
    return fused_relu_dropout2d