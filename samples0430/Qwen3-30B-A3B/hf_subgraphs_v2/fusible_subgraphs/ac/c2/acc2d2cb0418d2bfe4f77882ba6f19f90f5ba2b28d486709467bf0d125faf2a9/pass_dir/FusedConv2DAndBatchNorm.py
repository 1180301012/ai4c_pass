import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_8, in_9, tmp_4, tmp_5):
    tmp_6 = torch.conv2d(in_8, tmp_4, None, (1, 1), (3, 3), (1, 1), 300)
    tmp_7 = torch.conv2d(in_9, tmp_5, None, (1, 1), (4, 4), (1, 1), 300)
    return tmp_6, tmp_7

# Argument extraction function
def replacement_args(in_8, in_9, tmp_4, tmp_5):
    return (in_8, in_9, tmp_4, tmp_5)

# Triton kernel for fused convolutions
@triton.jit
def fused_conv_kernel(
    in1_ptr,  # [batch, in_c, H, W]
    in2_ptr,  # [batch, in_c, H, W]
    weights1_ptr,  # [out_c1, in_c, K1, K1]
    weights2_ptr,  # [out_c2, in_c, K2, K2]
    out_ptr,  # [batch, out_c1 + out_c2, H, W]
    batch_size, H, W, in_c, out_c1, out_c2, K1, K2, pad1, pad2,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Calculate the starting coordinate for the block
    pid = tl.program_id(0)
    h_start = (pid // (W // BLOCK_W)) * BLOCK_H
    w_start = (pid % (W // BLOCK_W)) * BLOCK_W
    
    # Calculate the block's boundaries
    h_end = min(h_start + BLOCK_H, H)
    w_end = min(w_start + BLOCK_W, W)
    
    # Process the spatial location range (h_start to h_end-1, w_start to w_end-1)
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            # Process the first convolution (out_c1 channels)
            for c1 in range(out_c1):
                val = 0.0
                for kh in range(K1):
                    for kw in range(K1):
                        h_in = h + kh - pad1
                        w_in = w + kw - pad1
                        if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                            # Load input value
                            input_val = tl.load(in1_ptr + h_in * W + w_in)
                            # Load weight value
                            weight_val = tl.load(weights1_ptr + c1 * (in_c * K1 * K1) + kh * K1 + kw)
                            val += input_val * weight_val
                tl.store(out_ptr + (h * W + w) * (out_c1 + out_c2) + c1, val)
            
            # Process the second convolution (out_c2 channels)
            for c2 in range(out_c2):
                val = 0.0
                for kh in range(K2):
                    for kw in range(K2):
                        h_in = h + kh - pad2
                        w_in = w + kw - pad2
                        if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
                            # Load input value
                            input_val = tl.load(in2_ptr + h_in * W + w_in)
                            # Load weight value
                            weight_val = tl.load(weights2_ptr + c2 * (in_c * K2 * K2) + kh * K2 + kw)
                            val += input_val * weight_val
                tl.store(out_ptr + (h * W + w) * (out_c1 + out_c2) + out_c1 + c2, val)

# Wrapper function
@torch.fx.wrap
def fused_conv_wrapper(in_8, in_9, tmp_4, tmp_5):
    # Extract shape information from the inputs
    batch_size, in_c, H, W = in_8.shape
    out_c1, _, K1, _ = tmp_4.shape
    out_c2, _, K2, _ = tmp_5.shape
    
    # The batch norm has a fixed padding for the kernel: (3, 3) and (4, 4)
    pad1 = (K1 - 1) // 2
    pad2 = (K2 - 1) // 2
    
    # Create an empty output tensor of the expected shape
    out = torch.empty((batch_size, out_c1 + out_c2, H, W), device=in_8.device, dtype=in_8.dtype)
    
    # Calculate the grid size
    grid = (H * W, )
    
    # Launch the kernel
    fused_conv_kernel[grid](
        in_8,
        in_9,
        tmp_4,
        tmp_5,
        out,
        batch_size,
        H,
        W,
        in_c,
        out_c1,
        out_c2,
        K1,
        K2,
        pad1,
        pad2,
        BLOCK_H=8,
        BLOCK_W=8
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv_wrapper