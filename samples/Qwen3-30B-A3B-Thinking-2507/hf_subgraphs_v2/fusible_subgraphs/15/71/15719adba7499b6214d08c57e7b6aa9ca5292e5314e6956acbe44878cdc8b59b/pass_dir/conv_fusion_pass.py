import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: conv2d → flatten(2) → transpose(1, 2)
def pattern(in_0, in_4, in_3, stride, padding, dilation, groups):
    conv = torch.conv2d(in_0, in_4, in_3, stride, padding, dilation, groups)
    tmp = conv.flatten(2)
    out = tmp.transpose(1, 2)
    return out

# Argument extraction function
# Extracts parameters for the custom kernel

def replacement_args(in_0, in_4, in_3, stride, padding, dilation, groups):
    return (in_0, in_4, in_3, stride, padding, dilation, groups)

# Optimized Triton kernel for fused convolution + reshape
@triton.jit
def conv_fusion_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, H_in, W_in,
    out_c, H_k, W_k,
    H_out, W_out,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    # Each block processes one (h, w) spatial position
    n = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # Process all output channels for this (h, w) position
    for out_c_idx in tl.range(out_c):
        val = tl.zeros((), dtype=tl.float32)
        
        # Convolution over kernel and input channel
        for kh in range(H_k):
            for kw in range(W_k):
                ih = h * stride_h - padding_h + kh * dilation_h
                iw = w * stride_w - padding_w + kw * dilation_w
                
                if ih >= 0 and ih < H_in and iw >= 0 and iw < W_in:
                    for c in range(C):
                        input_idx = n * C * H_in * W_in + c * H_in * W_in + ih * W_in + iw
                    
                for c in range(C):
                    input_idx = n * C * H_in * W_in + c * H_in * W_in + ih * W_in + iw
                    weight_idx = out_c_idx * C * H_k * W_k + c * H_k * W_k + kh * W_k + kw
                    
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    val += input_val * weight_val
        
        if bias_ptr:
            val += tl.load(bias_ptr + out_c_idx)
        
        # Store in flattened [N, H_out*W_out, out_c] layout
        output_idx = n * H_out * W_out * out_c + (h * W_out + w) * out_c + out_c_idx
        tl.store(output_ptr + output_idx, val)


# Kernel wrapper
@torch.fx.wrap

def conv_fusion(in_0, in_4, in_3, stride, padding, dilation, groups):
    # Parse input dimensions
    N, C, H_in, W_in = in_0.shape
    out_c, _, H_k, W_k = in_4.shape
    
    # Compute output spatial dimensions
    H_out = (H_in - H_k + 2 * padding[0]) // stride[0] + 1
    W_out = (W_in - W_k + 2 * padding[1]) // stride[1] + 1
    
    # Allocate output tensor: [N, H_out*W_out, out_c]
    output = torch.empty((N, H_out * W_out, out_c), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    grid = (N, H_out, W_out)
    conv_fusion_kernel[grid](
        in_0, in_4, in_3, output,
        N, C, H_in, W_in,
        out_c, H_k, W_k,
        H_out, W_out,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        BLOCK_SIZE_H=32,
        BLOCK_SIZE_W=32
    )
    
    return output

# Replacement function

def replacement_func():
    return conv_fusion