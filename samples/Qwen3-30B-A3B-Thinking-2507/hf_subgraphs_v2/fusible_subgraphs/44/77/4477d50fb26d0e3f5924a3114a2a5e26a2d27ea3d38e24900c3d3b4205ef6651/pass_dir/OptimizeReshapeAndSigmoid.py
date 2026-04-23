import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: conv2d → permute(0,2,3,1) → reshape(N, -1, M) → sigmoid
@torch.fx.wrap
def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(6, -1, 4)
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return (tmp_5,)

# Argument extraction function
# Returns the conv2d output
# (avoids recalculating conv2d in replacement)
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Triton kernel for sigmoid
@triton.jit
def sigmoid_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Stable sigmoid implementation
    s = 1.0 / (1.0 + tl.exp(-x))
    tl.store(output_ptr + offsets, s, mask=mask)

# Sigmoid wrapper using Triton
@torch.fx.wrap
def sigmoid_wrapper(input_tensor):
    n_elements = input_tensor.numel()
    block_size = 1024
    num_blocks = (n_elements + block_size - 1) // block_size
    output = torch.empty_like(input_tensor)
    sigmoid_kernel[(num_blocks,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
    )
    return output

# Optimized sequence: conv2d → reshape (no permute) → sigmoid
@torch.fx.wrap
def optimized_conv_sigmoid(conv2d):
    # Direct reshape: [B, M, H, W] → [B, H*W, M]
    B, M, H, W = conv2d.shape
    reshaped = conv2d.reshape(B, H * W, M)
    # Apply sigmoid via Triton kernel
    return sigmoid_wrapper(reshaped)

# Replacement function returning the optimized implementation
def replacement_func():
    return optimized_conv_sigmoid