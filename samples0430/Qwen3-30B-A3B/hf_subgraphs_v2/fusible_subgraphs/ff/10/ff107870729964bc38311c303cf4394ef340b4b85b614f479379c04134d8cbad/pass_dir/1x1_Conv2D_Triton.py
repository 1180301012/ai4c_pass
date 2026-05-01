import torch
import triton
import triton.language as tl


def pattern(in_10, in_8, in_7):
    conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_10, in_8, in_7):
    return (in_10, in_8, in_7)

@triton.jit
def conv1x1_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    c_in,
    c_out,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    # Input: [N, C_in, H, W] -> flattened to (N*H*W, C_in)
    # Weight: [C_out, C_in, 1, 1] -> (C_out, C_in)
    # Output: [N, C_out, H, W]
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate input row index (N*H*W index)
    row_idx = offsets // c_in
    # Calculate element within input row
    col_idx = offsets % c_in

    # Process one row of input
    for out_channel in range(c_out):
        # Load weight value
        weight = tl.load(w_ptr + out_channel * c_in + col_idx, mask=(col_idx < c_in))
        
        # Compute dot product for this output channel
        dot_product = tl.zeros((), dtype=tl.float32)
        for i in range(0, c_in, BLOCK_SIZE):
            # Load input chunk
            input_chunk = tl.load(
                x_ptr + row_idx * c_in + i,
                mask=(i + tl.arange(0, BLOCK_SIZE) < c_in) & mask,
                other=0.0
            )
            weight_chunk = tl.load(
                w_ptr + out_channel * c_in + i,
                mask=(i + tl.arange(0, BLOCK_SIZE) < c_in),
                other=0.0
            )
            dot_product += tl.dot(input_chunk, weight_chunk)

        # Add bias
        bias = tl.load(b_ptr + out_channel)
        out_val = dot_product + bias
        
        # Store output
        tl.store(out_ptr + row_idx * c_out + out_channel, out_val)

@torch.fx.wrap
def conv1x1_triton(x, w, b):
    batch, in_channels, H, W = x.shape
    out_channels, _, _, _ = w.shape

    n_elements = batch * H * W
    
    # Reshape w to [out_channels, in_channels]
    w_flat = w.squeeze(2).squeeze(2).reshape(out_channels, in_channels)
    
    # Reshape b to [out_channels]
    b_flat = b
    
    # Create output tensor
    out = torch.empty(batch, out_channels, H, W, dtype=x.dtype)
    
    # Determine grid size
    BLOCK_SIZE = 128
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    conv1x1_kernel[(grid,)](
        x_ptr=x,
        w_ptr=w_flat,
        b_ptr=b_flat,
        out_ptr=out,
        n_elements=n_elements,
        c_in=in_channels,
        c_out=out_channels,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return conv1x1_triton