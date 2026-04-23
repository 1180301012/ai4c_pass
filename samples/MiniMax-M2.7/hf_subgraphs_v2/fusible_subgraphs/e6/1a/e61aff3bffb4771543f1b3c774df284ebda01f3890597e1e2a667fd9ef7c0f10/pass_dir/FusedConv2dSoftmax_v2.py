import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HW: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for 1x1 conv + view + softmax pattern.
    
    The computation:
    1. 1x1 conv: conv[b, h, w] = sum_c(input[b, c, h, w] * weight[c]) + bias
    2. View: flat[hw] = conv[h, w] where hw = h*W + w
    3. Softmax: softmax[hw] = exp(flat[hw]) / sum_j(exp(flat[j]))
    4. Unsqueeze: output[h, w, 0] = softmax[hw]
    
    Grid: B * H * W programs (one per (b, h, w) position)
    """
    pid = tl.program_id(0)
    mask = pid < n_elements
    
    # Calculate (b, h, w) from program id
    b = pid // (H * W)
    h = (pid % (H * W)) // W
    w = (pid % (H * W)) % W
    
    # Compute base offset for this (b, h, w) in [B, C, H, W] tensor
    base_offset = b * C * H * W + h * W + w
    
    # Compute conv sum: sum over all C channels
    # For each channel c: input[b, c, h, w] * weight[c] + bias
    conv_sum = float(0.0)
    
    for c_idx in range(C):
        x = tl.load(input_ptr + base_offset + c_idx * H * W, mask=mask)
        w_val = tl.load(weight_ptr + c_idx, mask=mask)
        conv_sum += x * w_val
    
    # Add bias (broadcast scalar)
    bias_val = tl.load(bias_ptr)
    conv_sum += bias_val
    
    # Store the pre-softmax value (will be normalized externally)
    tl.store(output_ptr + pid, conv_sum, mask=mask)


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    B: tl.constexpr, HW: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel over the last dimension.
    
    Grid: B programs (one per batch)
    Each program processes one batch element with HW softmax values.
    """
    pid = tl.program_id(0)
    batch_offset = pid * HW
    
    # Load all HW values for this batch
    vals = tl.load(input_ptr + batch_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < HW)
    
    # Compute max for numerical stability
    max_val = tl.max(vals)
    
    # Compute exp(x - max) and sum
    exp_vals = tl.exp(vals - max_val)
    exp_sum = tl.sum(exp_vals)
    
    # Compute softmax
    softmax_vals = exp_vals / exp_sum
    
    # Store output
    tl.store(output_ptr + batch_offset + tl.arange(0, BLOCK_SIZE), softmax_vals, mask=tl.arange(0, BLOCK_SIZE) < HW)


@torch.fx.wrap
def fused_conv_softmax_kernel_wrapper(in_0, in_1, in_2, B, C, H, W):
    """
    Wrapper for the fused conv + view + softmax kernel.
    
    Args:
        in_0: bias tensor [1] or scalar
        in_1: weight tensor [1, C, 1, 1]
        in_2: input tensor [B, C, H, W]
        
    Returns:
        output tensor [B, 1, H*W, 1]
    """
    # Get dimensions
    HW = H * W
    
    # Reshape weight to [C] for easier access
    weight_1d = in_1.reshape(C)
    
    n_elements = B * H * W
    
    # Allocate buffer for pre-softmax values [B, H, W]
    pre_softmax = torch.empty((B, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel: one program per (b, h, w) position
    grid_conv = (n_elements,)
    BLOCK_SIZE = 1  # Each program processes one position
    
    fused_conv_softmax_kernel[grid_conv](
        in_2, weight_1d, in_0, pre_softmax,
        B, C, H, W, HW, n_elements, BLOCK_SIZE
    )
    
    # Now reshape and apply softmax using Triton
    # pre_softmax: [B, H, W] -> view: [B, 1, H*W]
    flat = pre_softmax.reshape(B, HW)
    
    # Allocate output buffer
    softmax_flat = torch.empty((B, HW), dtype=in_2.dtype, device=in_2.device)
    
    # Launch softmax kernel
    BLOCK_SIZE_SOFTMAX = min(4096, HW)
    grid_softmax = (B,)
    
    softmax_kernel[grid_softmax](
        flat, softmax_flat,
        B, HW, B, BLOCK_SIZE_SOFTMAX
    )
    
    # Reshape to [B, 1, H*W] and unsqueeze to [B, 1, H*W, 1]
    output = softmax_flat.reshape(B, 1, HW).unsqueeze(-1)
    
    return output


def pattern(in_0, in_1, in_2):
    """Match the conv + view + softmax + unsqueeze pattern for shape [4, 1, 192]."""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(4, 1, 192)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel = 5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement function."""
    # Get dimensions from input shapes
    B = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    return (in_0, in_1, in_2, B, C, H, W)


def replacement_func():
    """Return the replacement kernel wrapper."""
    return fused_conv_softmax_kernel_wrapper