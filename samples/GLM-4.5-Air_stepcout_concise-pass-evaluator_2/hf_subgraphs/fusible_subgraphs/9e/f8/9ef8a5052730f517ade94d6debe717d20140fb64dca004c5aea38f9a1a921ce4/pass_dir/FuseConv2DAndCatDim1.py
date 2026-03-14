import torch
import triton
import triton.language as tl

# Pattern matching - exactly matches the computation graph excluding cleanup operations
def pattern(in_0, in_1, in_2):
    """
    Matches Conv2D followed by Concat along dimension 1
    Pattern mirrors model.py exactly:
    - torch.conv2d with positional args: in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1
    - torch.cat with positional args: (tmp_1, in_2), 1
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.cat((tmp_1, in_2), 1)
    return (tmp_2,)  # Must return the same structure as original

# Argument extraction
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_cat_kernel(
    weight_ptr,           # [out_channels, in_channels, kH, kW]
    input_ptr,            # [batch, in_channels, H, W]
    concat_input_ptr,     # [batch, concat_channels, H, W]
    output_ptr,           # [batch, out_channels + concat_channels, H, W]
    batch, in_channels, H, W, out_channels, concat_channels,
    kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused Conv2D + Concat kernel - single position per program"""
    # Get program coordinates - each program handles one spatial position
    h = tl.program_id(0) * BLOCK_SIZE_M
    w = tl.program_id(1) * BLOCK_SIZE_N
    
    # Create mask for boundary check
    mask = (h < H) & (w < W)
    
    # Process each output channel
    for oc in range(out_channels):
        h_out, w_out = h, w
        
        # Compute convolution for this output position
        total = 0.0
        for ic in range(in_channels):
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate input coordinate with padding and dilation
                    ih = h_out * stride_h + dilation_h * kh - pad_h
                    iw = w_out * stride_w + dilation_w * kw - pad_w
                    
                    # Create input mask
                    input_mask = (0 <= ih) & (ih < H) & (0 <= iw) & (iw < W)
                    
                    if input_mask:
                        # Load input and weight
                        input_val = tl.load(
                            input_ptr + ic * H * W + ih * W + iw,
                            mask=input_mask,
                            other=0.0
                        )
                        
                        weight_idx = oc * in_channels * kH * kW + ic * kH * kW + kh * kW + kw
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True, other=0.0)
                        
                        total += input_val * weight_val
        
        # Store convolution result
        output_idx = oc * H * W + h_out * W + w_out
        tl.store(output_ptr + output_idx, total, mask=mask)
    
    # Process concatenation part
    for cc in range(concat_channels):
        h_out, w_out = h, w
        
        # Load concatenation value
        concat_idx = cc * H * W + h_out * W + w_out
        concat_val = tl.load(
            concat_input_ptr + concat_idx,
            mask=mask,
            other=0.0
        )
        
        # Store concatenation result
        output_idx = (out_channels + cc) * H * W + h_out * W + w_out
        tl.store(output_ptr + output_idx, concat_val, mask=mask)

@torch.fx.wrap
def fused_conv_cat(input, weight, concat_input):
    """Fused convolution + concatenation kernel wrapper"""
    # Get tensor shapes
    batch, in_channels, H, W = input.shape
    out_channels, _, kH, kW = weight.shape
    concat_channels = concat_input.shape[1]
    
    # Output shape: [batch, out_channels + concat_channels, H, W]
    output_shape = (batch, out_channels + concat_channels, H, W)
    output = torch.zeros(output_shape, dtype=input.dtype, device=input.device)
    
    # Choose block sizes based on input dimensions for optimal occupancy
    BLOCK_SIZE_M = 16  # Height block size
    BLOCK_SIZE_N = 16  # Width block size
    
    # Calculate grid size for 2D spatial grid
    grid_h = (H + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_w = (W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    if grid_h > 0 and grid_w > 0:
        fused_conv_cat_kernel[(grid_h, grid_w)](
            weight_ptr=weight,
            input_ptr=input,
            concat_input_ptr=concat_input,
            output_ptr=output,
            batch=batch, in_channels=in_channels, H=H, W=W,
            out_channels=out_channels, concat_channels=concat_channels,
            kH=kH, kW=kW,
            stride_h=1, stride_w=1, pad_h=1, pad_w=1,
            dilation_h=1, dilation_w=1,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
    
    return output

# Replacement function - returns the optimized kernel
def replacement_func():
    return fused_conv_cat