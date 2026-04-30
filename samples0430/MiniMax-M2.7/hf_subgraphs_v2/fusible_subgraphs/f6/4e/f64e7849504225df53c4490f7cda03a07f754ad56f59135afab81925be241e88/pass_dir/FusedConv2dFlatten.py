import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_flatten_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for 1x1 convolution followed by flatten.
    This avoids writing the intermediate [N, C_out, H, W] tensor.
    
    For a 1x1 conv with bias:
    out[n, oc, h, w] = bias[oc] + sum(ic, weight[oc, ic, 0, 0] * in[n, ic, h, w])
    
    Flatten(conv, 2) gives [N, C_out, H*W]
    """
    # Get program id for output index
    pid = tl.program_id(0)
    
    # Calculate output indices
    # Total output elements = N * C_out * H * W
    # We process BLOCK_SIZE elements per program
    n_elements = N * C_out * H * W
    
    # Calculate base offset for this program
    base = pid * BLOCK_SIZE
    
    # Create masks and offsets
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices
    # output layout: [n, oc, h*w] where h*w is flattened spatial dimension
    # We need to compute: n = offset // (C_out * H * W)
    #                     oc = (offset // (H * W)) % C_out
    #                     spatial_idx = offset % (H * W)
    
    # Flattened index to (n, oc, spatial_idx)
    n_idx = offsets // (C_out * H * W)
    oc_idx = (offsets // (H * W)) % C_out
    spatial_idx = offsets % (H * W)
    
    # Convert spatial_idx to (h, w)
    h_idx = spatial_idx // W
    w_idx = spatial_idx % W
    
    # Load bias for each output channel
    bias_vals = tl.load(bias_ptr + oc_idx, mask=mask)
    
    # Accumulate convolution result
    # sum over C_in dimension
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for ic_idx in range(C_in):
        # Load weight: [C_out, C_in, 1, 1]
        weight_offset = oc_idx * C_in + ic_idx
        w = tl.load(weight_ptr + weight_offset, mask=mask)
        
        # Load input: [N, C_in, H, W]
        # Input offset = n * C_in * H * W + ic * H * W + h * W + w
        in_offset = n_idx * C_in * H * W + ic_idx * H * W + h_idx * W + w_idx
        x = tl.load(in_ptr + in_offset, mask=mask)
        
        acc += w * x
    
    # Add bias
    result = acc + bias_vals
    
    # Store result [N, C_out, H*W]
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_conv2d_flatten_kernel_wrapper(bias, weight, in_tensor):
    """
    Wrapper for the fused conv2d + flatten kernel.
    
    Args:
        bias: Bias tensor [C_out]
        weight: Weight tensor [C_out, C_in, 1, 1]
        in_tensor: Input tensor [N, C_in, H, W]
    
    Returns:
        Output tensor [N, C_out, H*W]
    """
    N, C_in, H, W = in_tensor.shape
    C_out = weight.shape[0]
    
    # Output shape after flatten
    out = torch.empty((N, C_out, H * W), dtype=in_tensor.dtype, device=in_tensor.device)
    
    # Total number of output elements
    n_elements = N * C_out * H * W
    
    # Block size for Triton
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Execute kernel
    fused_conv2d_flatten_kernel[(num_programs,)](
        in_tensor,
        weight,
        bias,
        out,
        N,
        C_in,
        H,
        W,
        C_out,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match conv2d + flatten pattern.
    The computation is: conv2d(in_2, in_1, in_0, stride, padding, dilation, groups) then flatten(conv2d, 2)
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(conv2d, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the replacement function.
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the fused kernel wrapper function.
    """
    return fused_conv2d_flatten_kernel_wrapper