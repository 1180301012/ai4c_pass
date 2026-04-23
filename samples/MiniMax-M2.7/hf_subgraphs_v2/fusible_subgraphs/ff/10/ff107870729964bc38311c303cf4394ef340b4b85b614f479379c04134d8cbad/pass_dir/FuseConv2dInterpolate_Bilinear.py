import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_only_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    batch_size, out_channels, out_height, out_width,
    in_channels, in_height, in_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_elements = batch_size * out_channels * out_height * out_width
    output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < n_elements
    
    # Decode output position to get (batch, out_ch, h, w)
    tmp = output_idx
    w = tmp % out_width
    tmp = tmp // out_width
    h = tmp % out_height
    tmp = tmp // out_height
    ch = tmp % out_channels
    batch = tmp // out_channels
    
    # Initialize accumulator with bias
    acc = tl.load(bias_ptr + ch).to(tl.float32)
    
    # For 1x1 conv with stride 1, padding 0:
    # output[b, c_out, h, w] = sum(c_in) input[b, c_in, h, w] * weight[c_out, c_in, 0, 0]
    # Compute once and reuse h*in_width + w for all channels
    spatial_offset = h * in_width + w
    
    # Loop over input channels and accumulate
    for c_in in range(in_channels):
        # Input index: [b, c_in, h, w] -> linear = ((b * in_channels + c_in) * in_height + h) * in_width + w
        input_idx = ((batch * in_channels + c_in) * in_height + h) * in_width + w
        inp = tl.load(input_ptr + input_idx, mask=mask, other=0.0).to(tl.float32)
        
        # Weight index: [c_out, c_in, 0, 0] -> linear = (c_out * in_channels + c_in) * 1 * 1 + 0 * 1 + 0
        weight_idx = ch * in_channels + c_in
        weight_val = tl.load(weight_ptr + weight_idx).to(tl.float32)
        
        acc = acc + inp * weight_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc.to(tl.float16), mask=mask)


@torch.fx.wrap
def conv2d_interpolate_wrapper(in_10, in_8, in_7):
    """
    Conv2d operation using Triton kernel.
    in_10: input tensor (B, C_in, H, W)
    in_8: weight tensor (C_out, C_in, 1, 1)
    in_7: bias tensor (C_out,)
    """
    B, C_in, H, W = in_10.shape
    C_out = in_8.shape[0]
    
    # Allocate output
    output = torch.empty((B, C_out, H, W), dtype=in_10.dtype, device=in_10.device)
    
    # Grid configuration
    BLOCK_SIZE = 1024
    n_elements = B * C_out * H * W
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch conv2d kernel
    conv2d_only_kernel[(num_programs,)](
        in_10, in_8, in_7, output,
        B, C_out, H, W,
        C_in, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_10, in_8, in_7):
    """
    Pattern: conv2d only
    Matches the top path of the model.
    """
    conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)
    return conv2d


def replacement_args(in_10, in_8, in_7):
    return (in_10, in_8, in_7)


def replacement_func():
    return conv2d_interpolate_wrapper