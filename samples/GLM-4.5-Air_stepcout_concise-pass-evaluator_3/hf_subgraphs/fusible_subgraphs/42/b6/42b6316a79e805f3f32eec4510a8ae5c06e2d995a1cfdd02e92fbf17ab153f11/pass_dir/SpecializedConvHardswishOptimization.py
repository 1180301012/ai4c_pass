import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match Conv2D + Hardswish sequence"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(tmp_2, True)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fast_1x1_conv_kernel(
    input_ptr,  # [B, C_in, 1, 1]
    weight_ptr, # [C_out, C_in, 1, 1]
    bias_ptr,   # [C_out]
    output_ptr, # [B, C_out, 1, 1]
    B, C_in, C_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized for 1x1 convolutions where H=W=1
    
    # Program ID for output channel groups
    pid = tl.program_id(0)
    
    # Channel offsets within this program
    ch_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ch_mask = ch_offsets < C_out
    
    # Process each output channel
    for ch_idx in range(BLOCK_SIZE):
        ch = ch_offsets[ch_idx]
        if not ch_mask[ch_idx]:
            continue
            
        # Load bias
        bias_val = tl.load(bias_ptr + ch)
        
        # Load weight vectors for this output channel
        weights = tl.load(weight_ptr + ch * C_in + tl.arange(0, C_in))
        
        # Process all batch items
        for batch in range(B):
            # Load input vector for this batch
            inputs = tl.load(input_ptr + batch * C_in + tl.arange(0, C_in))
            
            # Vectorized dot product
            conv_result = tl.sum(weights * inputs) + bias_val
            
            # Hardswish activation: x * relu6(x + 3) / 6
            x_plus_3 = conv_result + 3.0
            relu6_result = tl.maximum(tl.minimum(x_plus_3, 6.0), 0.0)
            hardswish_result = conv_result * relu6_result / 6.0
            
            # Store result. Note: output is [B, C_out, 1, 1]
            tl.store(output_ptr + batch * C_out + ch, hardswish_result)

@torch.fx.wrap
def specialized_conv_hardswish(in_0, in_1, in_2):
    # Check if this is a 1x1 convolution case (H=W=1)
    B, C_in, H, W = in_2.shape
    
    # Only apply specialized kernel for 1x1 convolutions
    if H == 1 and W == 1:
        C_out = in_1.shape[0]
        
        # Create output tensor (still flattened later)
        output_shape = (B, C_out, H, W)
        output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
        
        # Optimal block size for 1x1 conv
        BLOCK_SIZE = min(32, C_out)
        num_programs = (C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Specialized kernel for 1x1 conv
        if C_out % BLOCK_SIZE == 0:
            fast_1x1_conv_kernel[(num_programs,)](
                in_2, in_1, in_0, output,
                B, C_in, C_out,
                BLOCK_SIZE
            )
        else:
            # Fallback to original kernel for non-divisible cases
            return fused_conv_hardswish(in_0, in_1, in_2)
    
    else:
        # For larger convolutions, use original fused convolution
        return fused_conv_hardswish(in_0, in_1, in_2)
    
    # Flatten to match original behavior
    return output.flatten(1, -1)

def replacement_func():
    return specialized_conv_hardswish