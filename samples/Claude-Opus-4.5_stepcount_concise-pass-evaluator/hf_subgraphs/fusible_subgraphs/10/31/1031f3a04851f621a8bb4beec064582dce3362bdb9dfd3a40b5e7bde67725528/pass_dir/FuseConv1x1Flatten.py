import torch
import triton
import triton.language as tl

# Pattern function - matches conv2d with 1x1 kernel followed by flatten
def pattern(in_0, in_1, in_2):
    # in_0: bias [C_out]
    # in_1: weight [C_out, C_in, 1, 1]
    # in_2: input [N, C_in, H, W]
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return (tmp_3,)

# Extract arguments for replacement
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C_IN': 32}, num_warps=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_C_IN': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C_IN': 32}, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C_IN': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C_IN': 32}, num_warps=8),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_C_IN': 64}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_C_IN': 32}, num_warps=8),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C_IN': 32}, num_warps=2),
    ],
    key=['N', 'C_in', 'C_out', 'HW'],
)
@triton.jit
def fused_conv1x1_flatten_kernel(
    x_ptr,           # Input tensor [N, C_in, H*W]
    w_ptr,           # Weight tensor [C_out, C_in]
    b_ptr,           # Bias tensor [C_out]
    o_ptr,           # Output tensor [N, C_out, H*W]
    N,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    HW,
    stride_x_n, stride_x_c,
    stride_o_n, stride_o_c,
    BLOCK_HW: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
):
    # Grid layout: (N * C_out, ceil(HW / BLOCK_HW))
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    # Decode n and c_out from pid_nc
    n = pid_nc // C_out
    c_out = pid_nc % C_out
    
    # Compute hw offsets for this block
    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    
    # Accumulate over input channels in blocks
    for c_in_start in range(0, C_in, BLOCK_C_IN):
        c_in_offs = c_in_start + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_offs < C_in
        
        # Load input[n, c_in_offs, hw_offs]: shape [BLOCK_C_IN, BLOCK_HW]
        x_offs = n * stride_x_n + c_in_offs[:, None] * stride_x_c + hw_offs[None, :]
        x_mask = c_in_mask[:, None] & hw_mask[None, :]
        x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
        
        # Load weight[c_out, c_in_offs]: shape [BLOCK_C_IN]
        w_offs = c_out * C_in + c_in_offs
        w = tl.load(w_ptr + w_offs, mask=c_in_mask, other=0.0)
        
        # Accumulate: acc[hw] += sum over c_in of x[c_in, hw] * w[c_in]
        acc += tl.sum(x * w[:, None], axis=0)
    
    # Add bias
    bias = tl.load(b_ptr + c_out)
    acc += bias
    
    # Store output[n, c_out, hw_offs]
    o_offs = n * stride_o_n + c_out * stride_o_c + hw_offs
    tl.store(o_ptr + o_offs, acc, mask=hw_mask)


@torch.fx.wrap
def fused_conv1x1_flatten_impl(bias, weight, x):
    # Get dimensions
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    HW = H * W
    
    # Reshape tensors - views, no copy
    x_flat = x.view(N, C_in, HW)
    w_flat = weight.view(C_out, C_in)
    out = torch.empty((N, C_out, HW), device=x.device, dtype=x.dtype)
    
    # Compute grid dimensions
    BLOCK_HW = 256
    grid = (N * C_out, triton.cdiv(HW, BLOCK_HW))
    
    # Launch kernel
    fused_conv1x1_flatten_kernel[grid](
        x_flat, w_flat, bias, out,
        N, C_in, C_out, HW,
        x_flat.stride(0), x_flat.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


def replacement_func():
    return fused_conv1x1_flatten_impl