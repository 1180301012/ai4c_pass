import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Pattern for Graph 0: batch size 1, view(1, 1, -1)
    tmp_2 = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(1, 1, -1)  # Exact match for Graph 0
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def optimized_softmax_kernel(
    in_ptr, out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized softmax kernel with numerical stability"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Shift and exponentiate
    shifted = x - max_val
    exp_x = tl.exp(shifted)
    
    # Compute sum
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Normalize
    softmax_out = exp_x / sum_exp
    
    # Store result
    tl.store(out_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def fused_conv2d_softmax_batch1(x, weight, bias):
    """Fused Conv2D + View + Softmax for batch size 1"""
    N, C, H, W = x.shape
    total_spatial = C * H * W
    
    # Step 1: Apply Conv2D operation (1x1 convolution)
    # Since weight is [1, C, 1, 1], this is essentially element-wise with bias addition
    # Handle bias correctly - if bias is scalar, broadcast it properly
    conv_output = x * weight.reshape(1, C, 1, 1)
    if bias.numel() == 1:
        # If bias is scalar [1], broadcast to all channels
        conv_output = conv_output + bias
    else:
        # If bias has proper channel dimension, add it correctly
        conv_output = conv_output + bias.reshape(1, C, 1, 1)
    
    # Step 2: Flatten spatial dimensions for softmax
    conv_flat = conv_output.reshape(N, C * H * W)
    
    # Step 3: Apply optimized softmax along spatial dimension
    out = torch.empty_like(conv_flat)
    
    # For batch size 1, we can optimize further
    optimized_softmax_kernel[(total_spatial + 1023) // 1024,](
        conv_flat[0], out[0],
        total_spatial,
        BLOCK_SIZE=1024,
    )
    
    # Reshape to match expected output format: [1, 1, spatial]
    return out.reshape(1, 1, C * H * W)

def replacement_func():
    return fused_conv2d_softmax_batch1