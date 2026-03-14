import torch
import triton
import triton.language as tl


# Optimized softmax kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel along the last dimension.
    Computes: exp(x_i) / sum_j(exp(x_j))
    Uses numerical stabilization: exp(x_i - max) / sum_j(exp(x_j - max))
    """
    # Each program handles one batch
    batch_idx = tl.program_id(0)
    
    # Offsets for this batch
    input_offset = batch_idx * n_elements
    output_offset = batch_idx * n_elements
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load all elements for this batch
    x = tl.load(input_ptr + input_offset + offs, mask=mask, other=0.0).to(tl.float32)
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Compute exp(x - max)
    exp_x = tl.exp(x - max_val)
    
    # Compute sum of exp values
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax_val = exp_x / sum_exp
    
    # Store result
    tl.store(output_ptr + output_offset + offs, softmax_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_conv_view_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    stride_input_n, stride_input_c, stride_input_h, stride_input_w,
    stride_weight_c,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Conv2d: input [N, C, H, W] @ weight [1, C, 1, 1] + bias -> [N, 1, H, W]
    2. View: reshape to [N, 1, H*W]
    
    Using vectorized loads and accumulates for better performance.
    """
    # Program ID gives us the flat index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Each output element corresponds to (batch, h, w)
    batch_idx = offs // (H * W)
    spatial_idx = offs % (H * W)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Compute conv for each element - start with zero
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over channels and accumulate
    # Process in chunks for better memory access patterns
    for c in range(C):
        # Compute input offsets
        input_base = batch_idx * stride_input_n
        input_offset = input_base + c * stride_input_c + h_idx * stride_input_h + w_idx * stride_input_w
        
        # Load input - vectorized load
        input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        # Load weight
        weight_val = tl.load(weight_ptr + c * stride_weight_c)
        
        # Multiply and accumulate in float32
        result = result + input_val * weight_val
    
    # Load bias and add at the end
    bias = tl.load(bias_ptr)
    result = result + bias
    
    # Store result
    tl.store(output_ptr + offs, result, mask=mask)


@torch.fx.wrap
def fused_conv_softmax_wrapper(input_tensor, weight_tensor, bias_tensor):
    """
    Wrapper function that launches the fused conv + view + softmax kernels.
    """
    N, C, H, W = input_tensor.shape
    
    # Allocate output tensor
    output = torch.empty((N, 1, H * W), dtype=torch.float32, device=input_tensor.device)
    
    # Intermediate buffer for conv output
    conv_output = torch.empty((N, 1, H * W), dtype=torch.float32, device=input_tensor.device)
    
    # First kernel: fused conv + view
    grid = (N,)
    
    # Weight is [1, C, 1, 1], need to flatten for Triton
    weight_flat = weight_tensor.squeeze().contiguous()
    
    fused_conv_view_kernel[grid](
        input_tensor, weight_flat, bias_tensor,
        conv_output,
        N, C, H, W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight_flat.stride(0),
        H * W,  # n_elements per batch
    )
    
    # Second kernel: softmax
    grid_softmax = (N,)
    
    softmax_kernel[grid_softmax](
        conv_output, output,
        H * W,  # n_elements per batch
    )
    
    return output


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: conv2d -> view -> softmax for graph with view(32, 1, -1)
    
    This is for graphs where the input batch size is 32.
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(32, 1, -1)
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_2, in_1, in_0)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_conv_softmax_wrapper