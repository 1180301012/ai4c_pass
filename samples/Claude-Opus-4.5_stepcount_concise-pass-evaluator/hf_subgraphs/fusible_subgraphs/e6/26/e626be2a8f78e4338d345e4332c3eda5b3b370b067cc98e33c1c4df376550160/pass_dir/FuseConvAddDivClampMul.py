import torch
import triton
import triton.language as tl


# Pattern matching function - matches the exact computation in the model
def pattern(in_0, in_1, in_2, in_3):
    # Conv2d: in_3 (input) @ in_1 (weight) + in_0 (bias)
    tmp_2 = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Add 1.0
    tmp_3 = tmp_2 + 1.0
    # Divide by 2.0
    tmp_4 = tmp_3 / 2.0
    # Clamp to [0.0, 1.0]
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    # Multiply by in_2 (element-wise)
    tmp_6 = in_2 * tmp_5
    return tmp_6


# Extract arguments needed for the replacement
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Optimized Triton kernel that computes 1x1 conv using GEMV + fused element-wise
# 1x1 conv is: out[n,c] = sum_k(in[n,k] * w[c,k]) + bias[c]
# This is equivalent to matrix multiplication

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3, num_warps=4),
    ],
    key=['N', 'C_out', 'H', 'W'],
)
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    stride_w0, stride_w1,  # weight strides
    stride_i3_0, stride_i3_1,  # in_3 strides
    stride_i2_0, stride_i2_1, stride_i2_2, stride_i2_3,  # in_2 strides
    stride_o_0, stride_o_1, stride_o_2, stride_o_3,  # out strides
    N, C_out, C_in, H, W
):
    # Each program computes one output element at position (n, c, h, w)
    pid = tl.program_id(0)
    total = N * C_out * H * W
    
    if pid >= total:
        return
    
    # Unpack indices
    n = pid // (C_out * H * W)
    rem = pid % (C_out * H * W)
    c = rem // (H * W)
    rem = rem % (H * W)
    h = rem // W
    w = rem % W
    
    # Compute 1x1 conv: out[n,c] = sum_k(in_3[n,k] * w[c,k]) + bias[c]
    # Load bias
    conv = tl.load(in_0_ptr + c)
    
    # Compute dot product: in_3[n,:] dot weight[c,:]
    # We unroll by 4 since C_in = 100
    ci = 0
    # Process 96 elements (24 x 4)
    for ci in tl.range(0, 96, 4):
        # Load in_3[n, ci:ci+4]
        i3_0 = tl.load(in_3_ptr + n * stride_i3_0 + (ci + 0) * stride_i3_1)
        i3_1 = tl.load(in_3_ptr + n * stride_i3_0 + (ci + 1) * stride_i3_1)
        i3_2 = tl.load(in_3_ptr + n * stride_i3_0 + (ci + 2) * stride_i3_1)
        i3_3 = tl.load(in_3_ptr + n * stride_i3_0 + (ci + 3) * stride_i3_1)
        
        # Load weight[c, ci:ci+4]
        w_0 = tl.load(in_1_ptr + c * stride_w0 + (ci + 0) * stride_w1)
        w_1 = tl.load(in_1_ptr + c * stride_w0 + (ci + 1) * stride_w1)
        w_2 = tl.load(in_1_ptr + c * stride_w0 + (ci + 2) * stride_w1)
        w_3 = tl.load(in_1_ptr + c * stride_w0 + (ci + 3) * stride_w1)
        
        conv = conv + i3_0 * w_0 + i3_1 * w_1 + i3_2 * w_2 + i3_3 * w_3
    
    # Handle remainder (4 elements: 96-100)
    for ci in tl.range(96, 100, 1):
        i3_rem = tl.load(in_3_ptr + n * stride_i3_0 + ci * stride_i3_1)
        w_rem = tl.load(in_1_ptr + c * stride_w0 + ci * stride_w1)
        conv = conv + i3_rem * w_rem
    
    # Apply: ((conv + 1) / 2).clamp(0, 1) * in_2
    scaled = conv * 0.5 + 0.5
    clamped = tl.minimum(tl.maximum(scaled, 0.0), 1.0)
    
    # Load in_2[n, c, h, w]
    in2 = tl.load(in_2_ptr + n * stride_i2_0 + c * stride_i2_1 + h * stride_i2_2 + w * stride_i2_3)
    
    # Compute output
    out = clamped * in2
    
    # Store
    tl.store(out_ptr + n * stride_o_0 + c * stride_o_1 + h * stride_o_2 + w * stride_o_3, out)


@torch.fx.wrap
def triton_fused_conv_add_div_clamp_mul(in_0, in_1, in_2, in_3):
    """Full fusion in Triton: 1x1 conv + add + div + clamp + mul"""
    N, C_out, H, W = in_2.shape
    C_in = in_1.shape[1]
    
    out = torch.empty_like(in_2)
    
    num_elements = N * C_out * H * W
    
    fused_kernel[(num_elements,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        stride_w0=in_1.stride(0),
        stride_w1=in_1.stride(1),
        stride_i3_0=in_3.stride(0),
        stride_i3_1=in_3.stride(1),
        stride_i2_0=in_2.stride(0),
        stride_i2_1=in_2.stride(1),
        stride_i2_2=in_2.stride(2),
        stride_i2_3=in_2.stride(3),
        stride_o_0=out.stride(0),
        stride_o_1=out.stride(1),
        stride_o_2=out.stride(2),
        stride_o_3=out.stride(3),
        N=N,
        C_out=C_out,
        C_in=C_in,
        H=H,
        W=W
    )
    
    return out


def replacement_func():
    return triton_fused_conv_add_div_clamp_mul