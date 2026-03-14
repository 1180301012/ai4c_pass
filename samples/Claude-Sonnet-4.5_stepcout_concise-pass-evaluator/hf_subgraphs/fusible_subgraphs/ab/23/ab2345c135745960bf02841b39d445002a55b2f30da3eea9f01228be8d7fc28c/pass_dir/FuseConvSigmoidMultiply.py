import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching the SE attention computation:
    conv2d -> sigmoid -> view -> multiply -> contiguous
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_se_attention_kernel(
    input_ptr,  # in_3: [1, C_in, 1, 1]
    weight_ptr,  # in_1: [C_out, C_in//groups, 1, 1]
    bias_ptr,  # in_0: [C_out]
    feature_ptr,  # in_2: [1, C_out, H, W]
    output_ptr,  # output: [1, C_out, H, W]
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    HW: tl.constexpr,
    groups: tl.constexpr,
    C_per_group_in: tl.constexpr,
    C_per_group_out: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fully fused SE attention kernel.
    Each program handles one channel and processes all spatial locations in blocks.
    """
    c = tl.program_id(0)
    
    if c >= C_out:
        return
    
    # Step 1: Compute grouped conv for this channel (ONCE)
    group_id = c // C_per_group_out
    c_in_start = group_id * C_per_group_in
    
    # Vectorized accumulation for conv
    conv_val = 0.0
    for c_in_rel in range(C_per_group_in):
        c_in = c_in_start + c_in_rel
        weight_idx = c * C_per_group_in + c_in_rel
        weight = tl.load(weight_ptr + weight_idx)
        input_val = tl.load(input_ptr + c_in)
        conv_val += weight * input_val
    
    # Add bias and apply sigmoid
    bias = tl.load(bias_ptr + c)
    scale = tl.sigmoid(conv_val + bias)
    
    # Step 2: Broadcast multiply across ALL spatial positions for this channel
    base_idx = c * HW
    
    # Process in blocks
    for block_start in range(0, HW, BLOCK_HW):
        hw_offsets = block_start + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < HW
        indices = base_idx + hw_offsets
        
        features = tl.load(feature_ptr + indices, mask=mask, other=0.0)
        output = features * scale
        tl.store(output_ptr + indices, output, mask=mask)


@torch.fx.wrap
def fused_se_attention(in_0, in_1, in_2, in_3):
    """
    Fused SE attention implementation with single kernel.
    Each thread block handles one complete channel.
    """
    # Get dimensions
    B, C_out, H, W = in_2.shape
    C_in = in_3.shape[1]
    groups = 4
    HW = H * W
    C_per_group_in = C_in // groups
    C_per_group_out = C_out // groups
    
    # Allocate output
    output = torch.empty_like(in_2)
    
    # Single kernel launch - 1D grid (one program per channel)
    grid = (C_out,)
    BLOCK_HW = 1024
    
    fused_se_attention_kernel[grid](
        in_3,
        in_1,
        in_0,
        in_2,
        output,
        C_in=C_in,
        C_out=C_out,
        HW=HW,
        groups=groups,
        C_per_group_in=C_per_group_in,
        C_per_group_out=C_per_group_out,
        BLOCK_HW=BLOCK_HW,
    )
    
    return output


def replacement_func():
    return fused_se_attention