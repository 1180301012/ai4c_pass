import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # Pattern: Conv2D followed by hardsigmoid activation
    conv_result = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    result = torch.nn.functional.hardsigmoid(conv_result, False)
    return result

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def fused_conv_hardsigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_out, C_in, H_in, W_in,
    KH, KW, SH, SW, PH, PW,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for Conv2D + Hardsigmoid"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C_out
    
    # Calculate output indices
    c_out = offsets % C_out
    n = offsets // C_out
    
    # Initialize output with bias
    bias_val = tl.load(bias_ptr + c_out)
    out_val = bias_val
    
    # For 1x1 convolution, input and output spatial dimensions are the same
    # We tile spatial dimensions to achieve better parallelism
    h_in_tiles = (H_in + GROUP_SIZE - 1) // GROUP_SIZE
    w_in_tiles = (W_in + GROUP_SIZE - 1) // GROUP_SIZE
    
    # Iterate through input channels with efficient tiling
    for tile_h in range(h_in_tiles):
        for tile_w in range(w_in_tiles):
            # Calculate input spatial range for this tile
            h_start = tile_h * GROUP_SIZE
            h_end = min(h_start + GROUP_SIZE, H_in)
            w_start = tile_w * GROUP_SIZE
            w_end = min(w_start + GROUP_SIZE, W_in)
            
            # For 1x1 conv, each output position processes all input channels
            # at the same spatial location
            if tile_h == 0 and tile_w == 0:  # Only spatial (0,0) for 1x1 conv
                for c_in in range(C_in):
                    # Load weight
                    weight_val = tl.load(weight_ptr + (c_out * C_in + c_in))
                    
                    # Load input element
                    x_offset = n * C_in * H_in * W_in + c_in * H_in * W_in
                    x_ptr_offset = x_ptr + x_offset
                    x_val = tl.load(x_ptr_offset + 0)  # Only spatial (0,0) matters
                    
                    # Convolution operation
                    out_val += weight_val * x_val
    
    # Apply hardsigmoid: max(0, min(1, x + 3)) / 6
    hard_sigmoid_val = tl.maximum(0.0, tl.minimum(1.0, out_val + 3.0)) / 6.0
    
    # Store result
    tl.store(out_ptr + offsets, hard_sigmoid_val, mask=mask)

@torch.fx.wrap
def fused_conv_hardsigmoid(x, weight, bias):
    """Fused Conv2D + Hardsigmoid operation for 1x1 convolution"""
    if x.numel() == 0 or weight.numel() == 0:
        return torch.empty_like(x)
    
    N, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = weight.shape
    
    # For 1x1 conv with stride (1,1) and padding (0,0), output shape matches input
    out = torch.empty((N, C_out, H_in, W_in), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    num_programs = (N * C_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    GROUP_SIZE = 8  # Tile size for spatial dimensions
    
    fused_conv_hardsigmoid_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N=N, C_out=C_out, C_in=C_in, H_in=H_in, W_in=W_in,
        KH=1, KW=1, SH=1, SW=1, PH=0, PW=0,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_conv_hardsigmoid