import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    in_1 += conv2d
    permuted = in_1.permute(0, 2, 1, 3)
    contiguous_result = permuted.contiguous()
    return (contiguous_result,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "groups4")


@triton.jit
def fused_conv2d_add_permute_kernel(
    in_1_ptr, in_2_ptr, weight_ptr, output_ptr,
    B, S, G, D,
    stride_in1_0, stride_in1_1, stride_in1_2, stride_in1_3,
    stride_in2_0, stride_in2_1, stride_in2_2, stride_in2_3,
    stride_w_0, stride_w_2,
    stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    PAD: tl.constexpr,
    K_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total = B * S * G * D
    mask = offsets < total
    
    # Decompose offset for output [B, S, G, D]
    d_idx = offsets % D
    g_idx = (offsets // D) % G
    s_idx = (offsets // (G * D)) % S
    b_idx = offsets // (S * G * D)
    
    # Compute depthwise conv: for output position (b, g, s, d)
    # conv_val = sum_{k=0}^{K_SIZE-1} padded_in_2[b, g, s-PAD+k, d] * weight[g, 0, k, 0]
    # With zero padding for out-of-range s values
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k_idx in range(K_SIZE):
        s_in = s_idx - PAD + k_idx
        valid_s = (s_in >= 0) & (s_in < S)
        # Load in_2[b, g, s_in, d]
        in_2_offset = b_idx * stride_in2_0 + g_idx * stride_in2_1 + s_in * stride_in2_2 + d_idx * stride_in2_3
        in_2_val = tl.load(in_2_ptr + in_2_offset, mask=(valid_s & mask), other=0.0).to(tl.float32)
        # Load weight[g, 0, k, 0] -> weight[g, k] with stride_w_0 and stride_w_2
        weight_offset = g_idx * stride_w_0 + k_idx * stride_w_2
        w_val = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0).to(tl.float32)
        acc += in_2_val * w_val
    
    # Load in_1[b, g, s, d]
    in_1_offset = b_idx * stride_in1_0 + g_idx * stride_in1_1 + s_idx * stride_in1_2 + d_idx * stride_in1_3
    in_1_val = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0).to(tl.float32)
    
    # result = in_1 + conv_val
    result = in_1_val + acc
    
    # Store to output[b, s, g, d] (permuted layout)
    out_offset = b_idx * stride_out_0 + s_idx * stride_out_1 + g_idx * stride_out_2 + d_idx * stride_out_3
    tl.store(output_ptr + out_offset, result, mask=mask)
    
    # Store back to in_1[b, g, s, d] (in-place modification)
    tl.store(in_1_ptr + in_1_offset, result, mask=mask)


def _fused_kernel_impl(in_0, in_1, in_2, groups):
    """Core implementation shared between groups4 and groups12"""
    B, G, S, D = in_1.shape
    assert G == groups
    
    # Allocate output tensor in permuted layout [B, S, G, D]
    output = torch.empty(B, S, G, D, dtype=in_1.dtype, device=in_1.device)
    
    total_elements = B * S * G * D
    BLOCK_SIZE = 1024
    # Ensure enough programs to cover all elements
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Weight shape: [G, 1, 65, 1], we access weight[g, 0, k, 0]
    # stride_w_0 = stride for dim 0 (groups), stride_w_2 = stride for dim 2 (kernel)
    
    fused_conv2d_add_permute_kernel[(num_programs,)](
        in_1_ptr=in_1, in_2_ptr=in_2, weight_ptr=in_0, output_ptr=output,
        B=B, S=S, G=G, D=D,
        stride_in1_0=in_1.stride(0), stride_in1_1=in_1.stride(1),
        stride_in1_2=in_1.stride(2), stride_in1_3=in_1.stride(3),
        stride_in2_0=in_2.stride(0), stride_in2_1=in_2.stride(1),
        stride_in2_2=in_2.stride(2), stride_in2_3=in_2.stride(3),
        stride_w_0=in_0.stride(0), stride_w_2=in_0.stride(2),
        stride_out_0=output.stride(0), stride_out_1=output.stride(1),
        stride_out_2=output.stride(2), stride_out_3=output.stride(3),
        PAD=32, K_SIZE=65,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def fused_conv2d_add_permute_dispatch(*args):
    route = args[-1]
    in_0, in_1, in_2 = args[:-1]
    if route == "groups4":
        return _fused_kernel_impl(in_0, in_1, in_2, groups=4)
    elif route == "groups12":
        return _fused_kernel_impl(in_0, in_1, in_2, groups=12)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return fused_conv2d_add_permute_dispatch