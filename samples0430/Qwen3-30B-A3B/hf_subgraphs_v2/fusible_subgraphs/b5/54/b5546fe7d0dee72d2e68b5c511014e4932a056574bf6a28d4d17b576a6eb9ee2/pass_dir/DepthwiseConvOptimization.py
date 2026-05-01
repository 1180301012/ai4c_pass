import torch
import triton
import triton.language as tl

# Pattern matching function
# Match: conv2d with groups = bias.numel()
def pattern(in_5, in_1, in_0):
    out = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), in_0.numel())
    return out

def replacement_args(in_5, in_1, in_0):
    # Extract required tensor shapes and strides
    C_out = in_0.numel()
    H, W = in_5.shape[2], in_5.shape[3]
    return (in_5, in_1, in_0, C_out, H, W)

@triton.jit
def depthwise_conv_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    C_out, H, W,
    input_stride1, input_stride2, input_stride3,
    weight_stride0, weight_stride2, weight_stride3,
    output_stride1, output_stride2, output_stride3,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Block indices
    blockIdx_h = tl.program_id(0)
    blockIdx_w = tl.program_id(1)
    blockIdx_c = tl.program_id(2)
    
    # Calculate output position
    h_start = blockIdx_h * BLOCK_H
    w_start = blockIdx_w * BLOCK_W
    c = blockIdx_c

    # Load weight for this channel into shared memory
    weights = tl.zeros((7, 7), dtype=tl.float32)
    for kh in tl.arange(0, 7):
        for kw in tl.arange(0, 7):
            weights[kh, kw] = tl.load(weight_ptr + c * weight_stride0 + kh * weight_stride2 + kw * weight_stride3)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Process input window
    for kh in tl.arange(0, 7):
        for kw in tl.arange(0, 7):
            ih = h_start + kh
            iw = w_start + kw
            
            # Load input with bounds checking
            input_val = tl.load(
                input_ptr + c * input_stride1 + ih * input_stride2 + iw * input_stride3,
                mask=(ih < H) & (iw < W),
                other=0.0
            )
            
            # Accumulate
            acc += input_val * weights[kh, kw]

    # Apply bias and store
    output = acc + tl.load(bias_ptr + c)
    out_ptr = output_ptr + c * output_stride1 + h_start * output_stride2 + w_start * output_stride3
    tl.store(
        out_ptr,
        output,
        mask=(h_start + tl.arange(0, BLOCK_H) < H) & (w_start + tl.arange(0, BLOCK_W) < W)
    )

@torch.fx.wrap
def depthwise_conv_wrapper(in_5, in_1, in_0, C_out, H, W):
    # Compute strides for the tensors
    _, C_in, H_in, W_in = in_5.shape
    _, _, Kh, Kw = in_1.shape
    
    input_stride1 = in_5.stride(1)
    input_stride2 = in_5.stride(2)
    input_stride3 = in_5.stride(3)

    weight_stride0 = in_1.stride(0)
    weight_stride2 = in_1.stride(2)
    weight_stride3 = in_1.stride(3)

    output_stride1 = in_5.stride(1)
    output_stride2 = in_5.stride(2)
    output_stride3 = in_5.stride(3)

    # Create output tensor
    output = torch.empty_like(in_5)

    # Define block size for spatial dimensions
    BLOCK_H = 8
    BLOCK_W = 8
    
    # Launch kernel
    num_blocks_h = (H + BLOCK_H - 1) // BLOCK_H
    num_blocks_w = (W + BLOCK_W - 1) // BLOCK_W
    
    depthwise_conv_kernel[(num_blocks_h, num_blocks_w, C_out), 1, 1](
        in_5, in_1, in_0,
        C_out, H, W,
        input_stride1, input_stride2, input_stride3,
        weight_stride0, weight_stride2, weight_stride3,
        output_stride1, output_stride2, output_stride3,
        BLOCK_H, BLOCK_W
    )
    
    return output

def replacement_func():
    return depthwise_conv_wrapper