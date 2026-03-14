import torch
import triton
import triton.language as tl


# Pattern matching: conv2d + hardsigmoid
# This is a common SE block pattern where conv output is immediately passed through hardsigmoid
def pattern(in_3, in_1, in_0):
    """
    Pattern: conv2d(in_3, in_1, in_0) -> hardsigmoid
    
    Only return the final output (hardsigmoid result) since the conv output
    is an intermediate value that's only used within this pattern.
    """
    # Conv2d: in_3 (B, C, 1, 1) @ in_1 (C_out, C_in, 1, 1) + in_0 (C_out)
    # The conv is depthwise with kernel size 1x1, so output is (B, C, 1, 1)
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Hardsigmoid on conv output
    tmp_3 = torch.nn.functional.hardsigmoid(tmp_2, False)
    
    # Only return tmp_3 (the hardsigmoid output) since tmp_2 is not observable
    # outside this pattern (it's immediately consumed by hardsigmoid)
    return tmp_3


def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)


# Optimized kernel: Fuses conv2d + hardsigmoid
# For 1x1 conv with groups=1, we can compute:
# output[b, c, 0, 0] = sum over c_in of (in_3[b, c_in, 0, 0] * weight[c, c_in, 0, 0]) + bias[c]
# Then apply hardsigmoid: max(0, min(1, (x + 3) / 6))
@triton.jit
def fused_conv_hardsigmoid_kernel(
    in_3_ptr,  # Input (B, C_in, 1, 1)
    in_1_ptr,  # Weight (C_out, C_in, 1, 1)
    in_0_ptr,  # Bias (C_out)
    output_ptr,  # Output (B, C_out, 1, 1)
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. 1x1 Conv2d (effectively a matrix multiply per batch element)
    2. Hardsigmoid activation
    """
    # Each program handles one output element: (b, c_out, 0, 0)
    # Grid: (B * C_out,)
    program_id = tl.program_id(0)
    b = program_id // C_out
    c_out = program_id % C_out
    
    # Load bias
    bias = tl.load(in_0_ptr + c_out)
    
    # Compute conv: sum over c_in of in_3[b, c_in] * weight[c_out, c_in]
    # Since spatial dimensions are 1x1, we just have scalar values
    conv_sum = 0.0
    for c_in in range(C_in):
        # in_3[b, c_in, 0, 0] is at offset b * C_in + c_in
        in_val = tl.load(in_3_ptr + b * C_in + c_in)
        
        # weight[c_out, c_in, 0, 0] is at offset c_out * C_in + c_in
        weight_val = tl.load(in_1_ptr + c_out * C_in + c_in)
        
        conv_sum += in_val * weight_val
    
    # Add bias
    conv_out = conv_sum + bias
    
    # Apply hardsigmoid: max(0, min(1, (x + 3) / 6))
    # This is equivalent to: (max(0, x) + 3) / 6, then min(1, result)
    # Or more precisely: clamp((x + 3) / 6, 0, 1)
    hsigmoid = (conv_out + 3.0) / 6.0
    hsigmoid = tl.maximum(0.0, hsigmoid)
    hsigmoid = tl.minimum(1.0, hsigmoid)
    
    # Store output[b, c_out, 0, 0]
    out_offset = b * C_out + c_out
    tl.store(output_ptr + out_offset, hsigmoid)


@torch.fx.wrap
def fused_conv_hardsigmoid_wrapper(in_3, in_1, in_0):
    """
    Wrapper function that launches the fused kernel.
    
    in_3: (B, C_in, 1, 1) - input tensor
    in_1: (C_out, C_in, 1, 1) - weight tensor
    in_0: (C_out,) - bias tensor
    """
    B, C_in, _, _ = in_3.shape
    C_out = in_1.shape[0]
    output = torch.empty((B, C_out, 1, 1), dtype=torch.float32, device=in_3.device)
    
    # Grid: one program per output element
    grid = (B * C_out,)
    
    # BLOCK_SIZE for the reduction loop
    BLOCK_SIZE = 1024
    
    fused_conv_hardsigmoid_kernel[grid](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        output_ptr=output,
        B=B,
        C_in=C_in,
        C_out=C_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_conv_hardsigmoid_wrapper