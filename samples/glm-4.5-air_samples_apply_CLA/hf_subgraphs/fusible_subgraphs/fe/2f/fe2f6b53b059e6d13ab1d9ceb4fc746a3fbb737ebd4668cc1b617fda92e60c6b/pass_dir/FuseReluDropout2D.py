import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the target computation
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel that fuses ReLU + Dropout2D
@triton.jit
def fused_relu_dropout_kernel(
    x_ptr,
    out1_ptr,      # dropout output
    out2_ptr,      # relu output
    n_elements,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Generate random mask for dropout - use offset as seed for reproducibility
    random_vals = tl.rand(offsets)
    dropout_mask = random_vals > dropout_p
    
    # Apply ReLU (this becomes part of both outputs)
    relu_out = tl.maximum(x, 0.0)
    
    # Apply dropout to create first output (tmp_1)
    dropout_out = relu_out * dropout_mask
    
    # Store both outputs
    tl.store(out1_ptr + offsets, dropout_out, mask=mask)
    tl.store(out2_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def fused_relu_dropout(x):
    # Get input tensor properties
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    
    # Create output tensors - same shape as inputs
    out1 = torch.empty_like(x)  # dropout output
    out2 = torch.empty_like(x)  # relu output
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_relu_dropout_kernel[(num_programs,)](
        x_ptr=x,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        dropout_p=0.1,  # dropout probability from original
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1, out2

# Replacement function
def replacement_func():
    return fused_relu_dropout