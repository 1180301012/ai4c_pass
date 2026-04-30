import torch
import triton
import triton.language as tl

# Pattern matching function - matches gelu followed by mean reduction
def pattern(in_0):
    """
    Match the pattern: gelu activation followed by mean over spatial dimensions.
    """
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def gelu_mean_kernel(
    in_ptr,
    out_gelu_ptr,
    out_mean_ptr,
    B,
    C,
    H,
    W,
    stride_B,
    stride_C,
    stride_H,
    stride_W,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get program ids for the output tensor dimensions
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate output pointers
    off_gelu = pid_b * stride_B + pid_c * stride_C + tl.arange(0, BLOCK_SIZE_B)[:, None] * stride_H + tl.arange(0, BLOCK_SIZE_C)[None, :] * stride_W
    off_mean = pid_b * C + pid_c
    
    # Load input data
    offs_h = tl.arange(0, H)
    offs_w = tl.arange(0, W)
    offs_b = tl.arange(0, BLOCK_SIZE_B)[:, None]
    offs_c = tl.arange(0, BLOCK_SIZE_C)[None, :]
    
    # Calculate input offsets - compute all H*W values
    in_offsets = (pid_b * stride_B + pid_c * stride_C + offs_h[None, :] * stride_H + offs_w[:, None] * stride_W)
    
    # Load data: shape (H, W)
    mask = (offs_h[None, :] < H) & (offs_w[:, None] < W)
    x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (x + 0.044715 * tl.pow(x, 3))))
    gelu_out = x * cdf
    
    # Store gelu output
    gelu_offsets = (pid_b * stride_B + pid_c * stride_C + offs_h[None, :] * stride_H + offs_w[:, None] * stride_W)
    tl.store(out_gelu_ptr + gelu_offsets, gelu_out, mask=mask)
    
    # Compute mean reduction over H*W
    sum_gelu = tl.sum(gelu_out, axis=0)  # Sum over H
    sum_gelu = tl.sum(sum_gelu, axis=0)  # Sum over W
    mean_val = sum_gelu / (H * W)
    
    # Store mean output (scalar)
    tl.store(out_mean_ptr + off_mean, mean_val)

@torch.fx.wrap
def fused_gelu_mean_kernel(in_0):
    """
    Fused kernel that computes both gelu and mean reduction in a single pass.
    """
    B, C, H, W = in_0.shape
    stride_B, stride_C, stride_H, stride_W = in_0.stride()
    
    # Create output tensors
    out_gelu = torch.empty_like(in_0)
    out_mean = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Grid configuration
    # Process (B, C) elements, each handling full H*W reduction
    grid = (B, C)
    
    # Kernel with fixed block sizes optimized for H=56, W=56
    gelu_mean_kernel[grid](
        in_0,
        out_gelu,
        out_mean,
        B, C, H, W,
        stride_B, stride_C, stride_H, stride_W,
        BLOCK_SIZE_B=H,
        BLOCK_SIZE_C=W,
    )
    
    return out_gelu, out_mean

def replacement_func():
    return fused_gelu_mean_kernel