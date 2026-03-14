import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: conv2d + stack([x], dim=0).sum(dim=0) + cat
    - stack([x], dim=0).sum(dim=0) is effectively a no-op
    - We eliminate stack+sum overhead
    """
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([tmp_2], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_2], 1)
    
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments for replacement function"""
    return (in_0, in_1, in_2, in_3)


# Use fixed-size blocks for the convolution
MAX_BLOCK = 256

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'NUM_WARPS': 4}),
        triton.Config({'BLOCK_OC': 128, 'NUM_WARPS': 4}),
        triton.Config({'BLOCK_OC': 256, 'NUM_WARPS': 8}),
    ],
    key=['B', 'H', 'W'],
)
@triton.jit
def fused_conv2d_cat_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    B, C_in, H, W,
    C_weight, C_other, C_out,
    stride_input, stride_weight, stride_bias, stride_other, stride_output,
    BLOCK_OC: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    """
    Fused kernel: 1x1 conv + cat
    Grid: B * H * W (one program per spatial position)
    """
    pid = tl.program_id(0)
    num_positions = B * H * W
    
    if pid >= num_positions:
        return
    
    b = pid // (H * W)
    rest = pid % (H * W)
    h = rest // W
    w = rest % W
    
    # Use fixed-size arange and mask
    oc_offsets = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_weight
    
    # Initialize output with bias
    conv_vals = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
    
    # Iterate over input channels
    for ic in range(C_in):
        # Load input value at (b, ic, h, w)
        input_val = tl.load(input_ptr + b * C_in * H * W + ic * H * W + h * W + w)
        
        # Load weight column ic: weight[0:BLOCK_OC, ic]
        weight_col = tl.load(weight_ptr + oc_offsets * C_in + ic, mask=oc_mask, other=0.0)
        
        # Accumulate: conv_vals += input_val * weight_col
        conv_vals = conv_vals + input_val * weight_col
    
    # Store conv result
    out_offsets = b * C_out * H * W + oc_offsets * H * W + h * W + w
    tl.store(output_ptr + out_offsets, conv_vals, mask=oc_mask)
    
    # Copy other input
    other_offsets = tl.arange(0, BLOCK_OC)
    other_mask = other_offsets < C_other
    other_vals = tl.load(other_ptr + b * C_other * H * W + 
                        other_offsets * H * W + h * W + w, mask=other_mask, other=0.0)
    
    out_offsets_other = b * C_out * H * W + (C_weight + other_offsets) * H * W + h * W + w
    tl.store(output_ptr + out_offsets_other, other_vals, mask=other_mask)


@torch.fx.wrap
def fused_conv2d_cat(in_0, in_1, in_2, in_3):
    """Fused 1x1 conv + cat using pure Triton"""
    B, C_in, H, W = in_3.shape
    C_weight = in_1.shape[0]
    C_other = in_2.shape[1]
    C_out = C_weight + C_other
    
    output = torch.empty((B, C_out, H, W), device=in_3.device, dtype=in_3.dtype)
    num_programs = B * H * W
    
    fused_conv2d_cat_kernel[(num_programs,)](
        in_3, in_1, in_0, in_2, output,
        B, C_in, H, W,
        C_weight, C_other, C_out,
        in_3.stride(0), in_1.stride(0), in_0.stride(0), in_2.stride(0), output.stride(0),
    )
    
    return output


def replacement_func():
    """Returns the replacement function"""
    return fused_conv2d_cat