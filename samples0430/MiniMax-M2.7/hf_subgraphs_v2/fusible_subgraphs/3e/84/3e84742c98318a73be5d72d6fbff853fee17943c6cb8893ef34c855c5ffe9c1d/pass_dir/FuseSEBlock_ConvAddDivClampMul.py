import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
    ],
    key=['C_in', 'C_out'],
)
@triton.jit
def se_fused_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    skip_ptr,
    out_ptr,
    B, C_in, C_out, H, W,
    add_val, div_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SE Block kernel - replaces entire SE block with single kernel.
    
    Performs:
    1. 1x1 Conv2D with bias
    2. Add scalar + divide scalar  
    3. Clamp to [0, 1]
    4. Multiply with skip connection
    
    For 1x1 conv with input [B, C_in, 1, 1], this is essentially:
    out[b, c] = sum_k(in[b, k] * weight[c, k]) + bias[c]
    
    Grid: (B, C_out) - each program handles one batch and one output channel
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + pid_c)
    
    # Compute 1x1 conv: out[c] = sum_k(in[b, k] * weight[c, k]) + bias[c]
    # For in [B, C_in, 1, 1] and weight [C_out, C_in, 1, 1]
    conv_result = 0.0
    
    # Loop over input channels for the GEMV
    for k in range(C_in):
        # in[b, k] is at in_ptr + pid_b * C_in + k
        in_val = tl.load(in_ptr + pid_b * C_in + k)
        # weight[c, k] is at weight_ptr + c * C_in + k
        w_val = tl.load(weight_ptr + pid_c * C_in + k)
        conv_result = conv_result + in_val * w_val
    
    # Add bias
    conv_result = conv_result + bias
    
    # Apply sigmoid-like: clamp((x + add_val) / div_val, 0, 1)
    activated = (conv_result + add_val) / div_val
    activated = tl.clamp(activated, 0.0, 1.0)
    
    # Multiply with skip connection: skip[b, c, h, w]
    # For each spatial location (h, w), multiply and store
    for h_idx in range(H):
        for w_idx in range(W):
            # Skip offset: b * C_out * H * W + c * H * W + h * W + w
            skip_offset = pid_b * C_out * H * W + pid_c * H * W + h_idx * W + w_idx
            skip_val = tl.load(skip_ptr + skip_offset)
            
            output = activated * skip_val
            
            # Store: out[b, c, h, w]
            out_offset = pid_b * C_out * H * W + pid_c * H * W + h_idx * W + w_idx
            tl.store(out_ptr + out_offset, output)





@torch.fx.wrap
def se_fused_wrapper(in_0, in_1, in_2, in_3, add_val=1.0, div_val=2.0):
    """
    Wrapper function for the fused SE block kernel.
    
    Args:
        in_0: bias tensor [C_out]
        in_1: weight tensor [C_out, C_in, 1, 1]
        in_2: skip connection tensor [B, C_out, H, W]
        in_3: input tensor [B, C_in, 1, 1]
        add_val: value to add before division (1.0 or 3.0)
        div_val: divisor (2.0 or 6.0)
    
    Returns:
        Output tensor [B, C_out, H, W]
    """
    B, C_in, H_in, W_in = in_3.shape  # [B, C_in, 1, 1]
    C_out = in_1.shape[0]  # [C_out, C_in, 1, 1]
    B_skip, C_out_skip, H_skip, W_skip = in_2.shape  # [B, C_out, H, W]
    
    assert H_in == 1 and W_in == 1, "This kernel expects spatial dimensions of 1"
    assert B == B_skip and C_out == C_out_skip, "Batch and channel dimensions must match"
    
    H, W = H_skip, W_skip
    
    # Allocate output
    out = torch.empty_like(in_2)  # Same shape as skip connection
    
    # Grid configuration
    grid = (B, C_out)
    
    # Launch kernel
    se_fused_kernel[grid](
        in_3, in_1, in_0, in_2, out,
        B, C_in, C_out, H, W,
        add_val, div_val,
    )
    
    return out


# Pattern matching function - matches the SE block computation
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the SE block pattern:
    conv2d(in_3, in_1, in_0) + add_val / div_val, clamp(0,1), mul(in_2)
    
    IMPORTANT: This must match the exact operations from model.py
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments for the replacement function.
    add_val=1.0 and div_val=2.0 are constants encoded in the pattern.
    """
    return (in_0, in_1, in_2, in_3, 1.0, 2.0)


def replacement_func():
    return se_fused_wrapper