import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    sigmoid = torch.sigmoid(conv)
    return sigmoid

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def conv_sigmoid_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_SIZE: tl.constexpr = 128
):
    # Thread index
    block_id = tl.program_id(0)
    start_idx = block_id * BLOCK_SIZE
    
    # Unpack channel, height, width
    c = start_idx // (out_height * out_width)
    remainder = start_idx % (out_height * out_width)
    h = remainder // out_width
    w = remainder % out_width
    
    # Calculate convolution output
    acc = 0.0
    for kh in range(1):
        for kw in range(1):
            # Convolution: 1x1 kernel
            in_val = tl.load(in_1_ptr + c * in_height * in_width + h * in_width + w)
            weight_val = tl.load(in_0_ptr + c * 1 * 1 + kh * 1 + kw)
            acc += in_val * weight_val
    
    # Apply sigmoid
    sigmoid_val = 1 / (1 + tl.exp(-acc))
    
    # Store result
    tl.store(out_ptr + c * out_height * out_width + h * out_width + w, sigmoid_val)

@torch.fx.wrap
def conv_sigmoid_wrapper(in_0, in_1):
    # Input shapes from weight_meta: in_0 [128, 960, 1, 1], in_1 [1, 960, 1, 4]
    B, C_in, H_in, W_in = in_1.shape
    C_out = in_0.shape[0]
    H_out = H_in  # 1 (since stride=1, padding=0)
    W_out = W_in  # 4
    
    # Output shape: [B, C_out, H_out, W_out]
    out = torch.empty((B, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device)
    
    # Calculate grid size
    total_elements = B * C_out * H_out * W_out
    BLOCK_SIZE = 128
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv_sigmoid_kernel[(num_blocks,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_channels=C_out,
        in_height=H_in,
        in_width=W_in,
        out_height=H_out,
        out_width=W_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return conv_sigmoid_wrapper