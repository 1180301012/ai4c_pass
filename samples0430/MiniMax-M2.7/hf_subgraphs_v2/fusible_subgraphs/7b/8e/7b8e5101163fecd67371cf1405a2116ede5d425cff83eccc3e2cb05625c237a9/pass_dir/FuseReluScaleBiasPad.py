import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu_scale_bias_pad_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    n_elements_padded,
   pad_l,
    pad_r,
    pad_t,
    pad_b,
    original_ndim,
    original_H,
    original_W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    - ReLU activation
    - Scale by scalar
    - Add bias
    - Pad output
    
    Args:
        x_ptr: Input tensor pointer
        scale_ptr: Scale scalar pointer
        bias_ptr: Bias scalar pointer
        output_ptr: Output tensor pointer
        n_elements: Original number of elements (before padding)
        n_elements_padded: Total number of elements (after padding)
        pad_l, pad_r: Left/right padding (typically 0, 1)
        pad_t, pad_b: Top/bottom padding (typically 0, 1)
        original_ndim: Number of spatial dimensions (2 for 4D tensor)
        original_H: Original height
        original_W: Original width
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input, scale, and bias
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load scale and bias as scalars
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    
    # Apply ReLU: max(0, x)
    x = tl.maximum(x, 0.0)
    
    # Apply scale and bias
    x = x * scale + bias
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias_pad(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + Scale + Bias + Pad operation.
    
    Args:
        x: Input tensor [B, C, H, W]
        scale: Scale factor, shape [1]
        bias: Bias term, shape [1]
    
    Returns:
        Padded tensor [B, C, H+1, W+1]
    """
    # Get original shape
    original_shape = x.shape
    ndim = x.ndim
    
    # Calculate padding parameters
    # For this pattern: pad (0, 1, 0, 1) means left=0, right=1, top=0, bottom=1
    pad_l, pad_r, pad_t, pad_b = 0, 1, 0, 1
    
    # Calculate output shape
    if ndim == 4:
        B, C, H, W = original_shape
        output_shape = (B, C, H + pad_t + pad_b, W + pad_l + pad_r)
    else:
        raise ValueError(f"Expected 4D input, got {ndim}D")
    
    # Number of elements
    n_elements = x.numel()
    n_elements_padded = B * C * (H + pad_t + pad_b) * (W + pad_l + pad_r)
    
    # Allocate output tensor
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Set up grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_relu_scale_bias_pad_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=scale,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        n_elements_padded=n_elements_padded,
        pad_l=pad_l,
        pad_r=pad_r,
        pad_t=pad_t,
        pad_b=pad_b,
        original_ndim=ndim,
        original_H=H,
        original_W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor) -> torch.Tensor:
    """
    Match the pattern: relu(in_2) * in_1 + in_0 with padding.
    
    Args:
        in_0: Bias tensor [1]
        in_1: Scale tensor [1]
        in_2: Input tensor [B, C, H, W]
    
    Returns:
        Padded output tensor
    """
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


def replacement_args(in_0: torch.Tensor, in_1: torch.Tensor, in_2: torch.Tensor):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1, in_2)


def replacement_func():
    """
    Return the fused implementation function.
    """
    return fused_relu_scale_bias_pad