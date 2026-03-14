import torch
import triton
import triton.language as tl

# Pattern to match: 1x1 conv followed by 2x2 avg pooling
def pattern(in_0, in_1):
    # in_0 is weight [C_out, C_in, 1, 1]
    # in_1 is input [N, C_in, H, W]
    conv_out = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    pool_out = torch.nn.functional.avg_pool2d(conv_out, 2, 2, 0, False, True, None)
    return (pool_out,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_COUT': 32, 'BLOCK_CIN': 64, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_COUT': 64, 'BLOCK_CIN': 32, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_COUT': 32, 'BLOCK_CIN': 32, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_COUT': 64, 'BLOCK_CIN': 64, 'BLOCK_HW': 16}, num_warps=4),
        triton.Config({'BLOCK_COUT': 128, 'BLOCK_CIN': 32, 'BLOCK_HW': 16}, num_warps=4),
        triton.Config({'BLOCK_COUT': 32, 'BLOCK_CIN': 128, 'BLOCK_HW': 16}, num_warps=4),
        triton.Config({'BLOCK_COUT': 16, 'BLOCK_CIN': 64, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_COUT': 64, 'BLOCK_CIN': 64, 'BLOCK_HW': 32}, num_warps=8),
    ],
    key=['N', 'C_in', 'C_out', 'H_out', 'W_out'],
)
@triton.jit
def fused_conv1x1_avgpool2x2_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H, W, C_out, H_out, W_out,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_o, stride_w_i,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_COUT: tl.constexpr,
    BLOCK_CIN: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel for 1x1 convolution followed by 2x2 average pooling.
    
    For efficiency, we first average pool the input (across 2x2 windows),
    then perform the 1x1 convolution. This is mathematically equivalent
    due to linearity of both operations.
    """
    # Program indices
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Output channel offsets
    cout_offsets = pid_cout * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    cout_mask = cout_offsets < C_out
    
    # Output spatial offsets (linearized h_out * W_out + w_out)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < (H_out * W_out)
    
    # Convert hw to h_out, w_out
    h_out = hw_offsets // W_out
    w_out = hw_offsets % W_out
    
    # Input spatial coordinates (before pooling)
    h_in_0 = h_out * 2
    h_in_1 = h_out * 2 + 1
    w_in_0 = w_out * 2
    w_in_1 = w_out * 2 + 1
    
    # Initialize accumulators [BLOCK_COUT, BLOCK_HW]
    acc = tl.zeros((BLOCK_COUT, BLOCK_HW), dtype=tl.float32)
    
    # Loop over input channels
    for cin_start in range(0, C_in, BLOCK_CIN):
        cin_offsets = cin_start + tl.arange(0, BLOCK_CIN)
        cin_mask = cin_offsets < C_in
        
        # Load weights [BLOCK_COUT, BLOCK_CIN]
        weight_ptrs = weight_ptr + cout_offsets[:, None] * stride_w_o + cin_offsets[None, :] * stride_w_i
        w = tl.load(weight_ptrs, mask=cout_mask[:, None] & cin_mask[None, :], other=0.0)
        
        # Load 4 input values for avg pooling: [BLOCK_CIN, BLOCK_HW]
        # Input shape: [N, C_in, H, W]
        base_ptr = input_ptr + pid_n * stride_in_n + cin_offsets[:, None] * stride_in_c
        
        # Position (h_in_0, w_in_0)
        ptr_00 = base_ptr + h_in_0[None, :] * stride_in_h + w_in_0[None, :] * stride_in_w
        x00 = tl.load(ptr_00, mask=cin_mask[:, None] & hw_mask[None, :], other=0.0)
        
        # Position (h_in_0, w_in_1)
        ptr_01 = base_ptr + h_in_0[None, :] * stride_in_h + w_in_1[None, :] * stride_in_w
        x01 = tl.load(ptr_01, mask=cin_mask[:, None] & hw_mask[None, :], other=0.0)
        
        # Position (h_in_1, w_in_0)
        ptr_10 = base_ptr + h_in_1[None, :] * stride_in_h + w_in_0[None, :] * stride_in_w
        x10 = tl.load(ptr_10, mask=cin_mask[:, None] & hw_mask[None, :], other=0.0)
        
        # Position (h_in_1, w_in_1)
        ptr_11 = base_ptr + h_in_1[None, :] * stride_in_h + w_in_1[None, :] * stride_in_w
        x11 = tl.load(ptr_11, mask=cin_mask[:, None] & hw_mask[None, :], other=0.0)
        
        # Average pool: [BLOCK_CIN, BLOCK_HW]
        pooled = (x00 + x01 + x10 + x11) * 0.25
        
        # Matrix multiply: [BLOCK_COUT, BLOCK_CIN] @ [BLOCK_CIN, BLOCK_HW] -> [BLOCK_COUT, BLOCK_HW]
        acc += tl.dot(w, pooled)
    
    # Store results
    out_base = output_ptr + pid_n * stride_out_n
    out_ptrs = out_base + cout_offsets[:, None] * stride_out_c + h_out[None, :] * stride_out_h + w_out[None, :] * stride_out_w
    tl.store(out_ptrs, acc, mask=cout_mask[:, None] & hw_mask[None, :])


@torch.fx.wrap
def fused_conv1x1_avgpool2x2(weight, input):
    """
    Fused 1x1 convolution + 2x2 average pooling.
    
    Args:
        weight: [C_out, C_in, 1, 1]
        input: [N, C_in, H, W]
    
    Returns:
        output: [N, C_out, H//2, W//2]
    """
    N, C_in, H, W = input.shape
    C_out = weight.shape[0]
    H_out = H // 2
    W_out = W // 2
    
    # Make sure input is contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    
    # Squeeze weight to [C_out, C_in] since it's 1x1
    weight_2d = weight.view(C_out, C_in)
    
    output = torch.empty((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)
    
    # Grid dimensions
    grid = lambda meta: (
        N,
        triton.cdiv(C_out, meta['BLOCK_COUT']),
        triton.cdiv(H_out * W_out, meta['BLOCK_HW']),
    )
    
    fused_conv1x1_avgpool2x2_kernel[grid](
        input, weight_2d, output,
        N, C_in, H, W, C_out, H_out, W_out,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight_2d.stride(0), weight_2d.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return (output,)


def replacement_func():
    return fused_conv1x1_avgpool2x2