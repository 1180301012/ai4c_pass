import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_1, in_0):
    return (in_0, in_1)

BLOCK_SIZE_H = 32
BLOCK_SIZE_W = 32

@triton.jit
def fused_conv_mean_kernel(
    in_1_ptr,
    in_0_ptr,
    out_ptr,
    sum_ptr,
    B, C_out, C_in, H_in, W_in, H_out, W_out, K,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_H,
    BLOCK_SIZE_W
):
    # Get block indices
    block_id = tl.program_id(0)
    b = block_id // C_out
    c = block_id % C_out
    
    start_h = tl.program_id(1) * BLOCK_SIZE_H
    start_w = tl.program_id(2) * BLOCK_SIZE_W
    end_h = min(start_h + BLOCK_SIZE_H, H_out)
    end_w = min(start_w + BLOCK_SIZE_W, W_out)
    
    # Initialize sum for this (b, c)
    sum_val = tl.zeros((), dtype=tl.float16)
    
    # Loop over spatial tile
    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            # Compute conv2d value for (b, c, h, w)
            val = 0.0
            for i in range(K):
                for j in range(K):
                    for k in range(C_in):
                        # Calculate input coordinates
                        h_in = h * stride_h + i * dilation_h - padding_h
                        w_in = w * stride_w + j * dilation_w - padding_w
                        if h_in < 0 or h_in >= H_in or w_in < 0 or w_in >= W_in:
                            continue
                        in1_val = tl.load(in_1_ptr + b * C_in * H_in * W_in + k * H_in * W_in + h_in * W_in + w_in)
                        in0_val = tl.load(in_0_ptr + c * C_in * K * K + k * K * K + i * K + j)
                        val += in1_val * in0_val
            # Store conv2d value
            tl.store(out_ptr + b * C_out * H_out * W_out + c * H_out * W_out + h * W_out + w, val)
            # Accumulate for mean
            sum_val += val
    
    # Write sum to sum tensor
    sum_index = b * C_out + c
    tl.store(sum_ptr + sum_index, sum_val)

@torch.fx.wrap
def fused_conv_mean(in_0, in_1):
    # Determine dimensions
    B, C_in, H_in, W_in = in_1.shape
    C_out, _, K, _ = in_0.shape
    
    # Calculate output dimensions
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 1, 1
    dilation_h, dilation_w = 1, 1
    H_out = (H_in + 2 * padding_h - dilation_h * (K - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * padding_w - dilation_w * (K - 1) - 1) // stride_w + 1
    
    # Create output tensors
    out = torch.empty((B, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device)
    sum_tensor = torch.empty((B, C_out), dtype=in_1.dtype, device=in_1.device)
    
    # Grid dimensions
    grid_y = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_z = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    fused_conv_mean_kernel[
        (B * C_out, grid_y, grid_z)
    ](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_ptr=out,
        sum_ptr=sum_tensor,
        B=B, C_out=C_out, C_in=C_in, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out, K=K,
        stride_h=stride_h, stride_w=stride_w,
        padding_h=padding_h, padding_w=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    # Compute mean
    mean_tensor = sum_tensor / (H_out * W_out)
    mean_tensor = mean_tensor.view(B, C_out, 1, 1)
    
    return out, mean_tensor

def replacement_func():
    return fused_conv_mean