import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches conv3d -> flatten(2) -> transpose(1, 2)
def pattern(in_0, in_1, in_3):
    conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    tmp_4 = conv3d.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    return tmp_5

# Argument extraction function
# Returns inputs needed for the optimized kernel

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

# Triton kernel for fused operation
@triton.jit
def fused_conv3d_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                       batch, in_c, time, height, width,
                       out_c, out_time, out_height, out_width,
                       stride_t, stride_h, stride_w,
                       BLOCK_OUT_C: tl.constexpr, BLOCK_SEQ: tl.constexpr):
    # Block IDs
    seq_idx = tl.program_id(0) * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    out_c_idx = tl.program_id(1) * BLOCK_OUT_C + tl.arange(0, BLOCK_OUT_C)

    # Compute sequence index
    seq_idx = seq_idx[:, None]  # shape (BLOCK_SEQ, 1)
    out_c_idx = out_c_idx[None, :]  # shape (1, BLOCK_OUT_C)
    
    # Convert sequence index to (t, h, w) indices
    t_idx = seq_idx // (out_height * out_width)
    h_idx = (seq_idx % (out_height * out_width)) // out_width
    w_idx = seq_idx % out_width

    # Compute the input position for convolution
    input_t = t_idx * stride_t
    input_h = h_idx * stride_h
    input_w = w_idx * stride_w

    # Compute output position in (batch, seq, out_c)
    output_idx = (0 * out_time * out_height * out_width * out_c) + \
                (seq_idx * out_c) + out_c_idx

    # Accumulate for convolution
    acc = tl.zeros((BLOCK_SEQ, BLOCK_OUT_C), dtype=tl.float32)
    for ic in range(in_c):
        for kt in range(2):  # kernel time
            for kh in range(16):  # kernel height
                for kw in range(16):  # kernel width
                    # Input indices
                    input_t_idx = input_t + kt
                    input_h_idx = input_h + kh
                    input_w_idx = input_w + kw
                    mask = (input_t_idx < time) & (input_h_idx < height) & (input_w_idx < width)

                    # Load input and weight
                    input_val = tl.load(
                        input_ptr + (
                            0 * in_c * time * height * width + 
                            ic * time * height * width + 
                            input_t_idx * height * width + 
                            input_h_idx * width + 
                            input_w_idx
                        ),
                        mask=mask,
                        other=0.0
                    )
                    weight_val = tl.load(
                        weight_ptr + (
                            out_c_idx * in_c * 2 * 16 * 16 + 
                            ic * 2 * 16 * 16 + 
                            kt * 16 * 16 + 
                            kh * 16 + 
                            kw
                        ),
                        mask=(kt < 2) & (kh < 16) & (kw < 16),
                        other=0.0
                    )
                    acc += input_val * weight_val

    # Add bias
    bias_val = tl.load(
        bias_ptr + out_c_idx,
        mask=(out_c_idx < out_c),
        other=0.0
    )
    acc += bias_val

    # Store output
    tl.store(output_ptr + output_idx, acc, mask=(out_c_idx < out_c) & (seq_idx < out_time * out_height * out_width))

# Kernel wrapper
@torch.fx.wrap
def fused_conv3d_transpose(in_0, in_1, in_3):
    batch, in_c, time, height, width = in_3.shape
    out_c = in_1.shape[0]
    
    # Calculate output dimensions
    out_time = (time - 2) // 2 + 1
    out_height = (height - 16) // 16 + 1
    out_width = (width - 16) // 16 + 1
    seq_len = out_time * out_height * out_width
    
    # Initialize output
    output = torch.empty((batch, seq_len, out_c), dtype=in_3.dtype, device=in_3.device)
    
    # Configure kernel grid
    grid = (
        (seq_len + 63) // 64,  # num blocks for sequence
        (out_c + 63) // 64       # num blocks for output channels
    )
    
    # Launch kernel
    fused_conv3d_kernel[grid](
        in_3, 
        in_1, 
        in_0, 
        output,
        batch, in_c, time, height, width,
        out_c, out_time, out_height, out_width,
        2, 16, 16,
        BLOCK_OUT_C=64,
        BLOCK_SEQ=64
    )
    
    return output

# Replacement function

def replacement_func():
    return fused_conv3d_transpose