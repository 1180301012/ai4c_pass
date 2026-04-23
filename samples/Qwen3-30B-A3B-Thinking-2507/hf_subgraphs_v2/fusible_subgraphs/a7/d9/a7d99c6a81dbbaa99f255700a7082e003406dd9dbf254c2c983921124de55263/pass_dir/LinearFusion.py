import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches the linear operation followed by view and transpose
def pattern(in_3, in_1, in_0):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    return tmp_6

# Argument extraction function
def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Triton kernel for fused linear + view + transpose
@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a segment of output vector (size BLOCK_SIZE)
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load bias
    bias = tl.load(bias_ptr + offsets, mask=mask)
    out = bias

    # Load input vector
    input_vec = tl.load(input_ptr + tl.arange(0, n_elements), mask=tl.arange(0, n_elements) < n_elements)

    # Compute dot product for each output element
    for k in range(n_elements):
        x = input_vec[k]
        weight_vals = tl.load(weight_ptr + k * n_elements + offsets, mask=mask, other=0.0)
        out += x * weight_vals

    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

# Kernel wrapper that handles output shape
@torch.fx.wrap
def linear_wrapper(in_3, in_1, in_0):
    # Reshape input to [512] for computation
    in_3_flat = in_3.view(-1)
    n_elements = 512

    # Create output tensor with desired shape [1,8,1,64]
    out = torch.empty(1, 8, 1, 64, dtype=in_3.dtype, device=in_3.device)
    out_flat = out.view(-1)  # [512]

    # Configure kernel launch
    BLOCK_SIZE = 64
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    linear_kernel[(num_blocks,)](
        in_3_flat,
        in_1,
        in_0,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Replacement function returns the wrapper
def replacement_func():
    return linear_wrapper