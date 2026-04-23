import torch
import triton
import triton.language as tl


def pattern(in_2, in_0, in_1):
    """
    Match the pattern: conv2d -> iadd -> permute -> contiguous
    This fuses multiple operations into a single kernel.
    Note: Using += (iadd) to match the in-place add in the model.
    """
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_2, tmp_0, None, (1, 1), (32, 0), (1, 1), 4)
    tmp_0 = None
    in_1 += tmp_1
    tmp_2 = in_1
    tmp_1 = None
    tmp_3 = tmp_2.permute(0, 2, 1, 3)
    tmp_2 = None
    tmp_4 = tmp_3.contiguous()
    tmp_3 = None
    return tmp_4


def replacement_args(in_2, in_0, in_1):
    return (in_2, in_0, in_1)


@triton.jit
def fused_conv2d_add_permute_kernel(
    # Conv2d inputs
    input_ptr,
    weight_ptr,
    bias_ptr,
    # Add input (will be added to conv output)
    add_ptr,
    # Output
    output_ptr,
    # Conv2d parameters
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    # Tensor shapes
    B: tl.constexpr,
    C_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    C_out: tl.constexpr,
    # Output shape after permute [B, H_after, C_after, W_after]
    B_out: tl.constexpr,
    H_after: tl.constexpr,
    C_after: tl.constexpr,
    W_after: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: conv2d + add + permute(0, 2, 1, 3) + contiguous
    
    This kernel:
    1. Performs depthwise conv2d
    2. Adds the result to the input tensor
    3. Permutes dimensions from [B, C, H, W] to [B, H, C, W]
    4. Writes output in contiguous memory layout
    
    Input shapes:
    - input: [B, C_in, H_in, W_in] (value_layer)
    - weight: [C_out, 1, kH, kW] (conv weight)
    - add: [B, C_out, H_out, W_out] (context_layer)
    
    Output shape: [B, H_out, C_out, W_out] after permute
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B_out * H_after * C_after * W_after
    
    # Output position [b, h, c, w]
    b_out = offsets // (H_after * C_after * W_after)
    rem = offsets % (H_after * C_after * W_after)
    h_out = rem // (C_after * W_after)
    rem = rem % (C_after * W_after)
    c_out = rem // W_after
    w_out = rem % W_after
    
    # After permute(0, 2, 1, 3), output[b, h, c, w] = add[b, c, h, w]
    # (assuming add has shape [B, C, H, W] matching conv output)
    add_idx = b_out * C_after * H_after * W_after + c_out * H_after * W_after + h_out * W_after + w_out
    
    # Load add value
    add_val = tl.load(add_ptr + add_idx, mask=mask, other=0.0)
    
    # For conv2d, we need to compute the contribution from input
    # The output channel c_out belongs to group c_out / (C_out / groups)
    # Within the group, the input channel is c_out % (C_out / groups)
    
    # Conv output for position [b, c_out, h_out, w_out] = sum over kernel
    conv_val = 0.0
    
    # Kernel size is 65x1 for this model
    kH = 65
    kW = 1
    
    # For depthwise conv with groups:
    # - Each output channel is computed independently
    # - Input channel = output channel (for depthwise)
    
    # In this case, weight has shape [C_out, 1, kH, kW]
    # For output channel c_out:
    # - weight[c_out, 0, kh, kw] is the kernel weight
    # - input[b, c_out, h_in, w_in] where h_in = h_out - padding + kh * dilation
    
    # Calculate input position for this output
    # h_out = h_in * stride_h - padding_h + kh * dilation_h
    # So h_in = (h_out + padding_h - kh * dilation_h) / stride_h
    
    stride_h_val = stride_h
    padding_h_val = padding_h
    dilation_h_val = dilation_h
    
    for kh in range(kH):
        # h_in = (h_out + padding_h - kh * dilation_h) / stride_h
        h_in = (h_out + padding_h_val - kh * dilation_h_val) // stride_h_val
        
        # Skip invalid positions
        if h_in < 0 or h_in >= H_in:
            continue
            
        # w_out = w_in * stride_w - padding_w + kw * dilation_w
        # Since kW=1 and dilation_w=1, padding_w must handle alignment
        w_in = (w_out + padding_w - 0) // stride_w
        
        if w_in < 0 or w_in >= W_in:
            continue
        
        # Input index: [b, c, h_in, w_in]
        # For depthwise: c = c_out
        input_idx = b_out * C_in * H_in * W_in + c_out * H_in * W_in + h_in * W_in + w_in
        
        # Weight index: [c_out, 0, kh, 0]
        weight_idx = c_out * kH * kW + kh * kW + 0
        
        # Load values
        input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=mask, other=0.0)
        
        conv_val += input_val * weight_val
    
    # Add conv output to add value
    result = add_val + conv_val
    
    # Store to output (already permuted and contiguous)
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_conv2d_add_permute_wrapper(in_2, in_0, in_1):
    """
    Wrapper for the fused conv2d + add + permute + contiguous kernel.
    """
    # in_0: weight [C_out, 1, kH, kW]
    # in_2: input [B, C_in, H_in, W_in]
    # in_1: add [B, C_out, H_out, W_out]
    
    B, C_out, H_out, W_out = in_1.shape
    C_in = in_2.shape[1]
    H_in = in_2.shape[2]
    W_in = in_2.shape[3]
    
    # Output shape after permute(0, 2, 1, 3): [B, H_out, C_out, W_out]
    output = torch.empty((B, H_out, C_out, W_out), dtype=in_1.dtype, device=in_1.device)
    
    # Grid configuration
    BLOCK_SIZE = 512
    n_elements = B * H_out * C_out * W_out
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_add_permute_kernel[(num_programs,)](
        input_ptr=in_2,
        weight_ptr=in_0,
        bias_ptr=0,  # No bias in original conv2d
        add_ptr=in_1,
        output_ptr=output,
        stride_h=1,
        stride_w=1,
        padding_h=32,
        padding_w=0,
        dilation_h=1,
        dilation_w=1,
        groups=4,
        B=B,
        C_in=C_in,
        H_in=H_in,
        W_in=W_in,
        C_out=C_out,
        B_out=B,
        H_after=H_out,
        C_after=C_out,
        W_after=W_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_conv2d_add_permute_wrapper