import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: softmax(in_1, dim=1) * in_0, then sum(dim=1)
    This corresponds to a weighted sum where softmax provides the weights.
    """
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_softmax_mul_sum_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    N, C, H, W,
    stride_in_0, stride_in_1,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: sum(in_0 * softmax(in_1, dim=1), dim=1)
    
    Inputs:
    - in_0: [B, C, H, W] = [1, 2, 256, 32, 32] or [1, 2, 256, 8, 8]
    - in_1: [B, C, H, W] = [1, 2, 256, 1, 1]
    Output: [B, H, W] = [1, 256, 32, 32] or [1, 256, 8, 8]
    
    Since in_1 has spatial dimensions 1x1, the softmax is computed per (B, H, W) position
    over the C (channel) dimension.
    """
    # Each program processes a single (h, w) position for all channels
    # Grid: (B * H * W)
    pid = tl.program_id(0)
    
    # Calculate batch, h, w indices
    # Assuming B=1, grid is H*W
    b = 0  # batch index always 0
    h = pid // W
    w = pid % W
    
    # Compute offset for output and inputs
    # Output shape: [B, H, W] -> flatten to [B*H*W]
    out_offset = pid
    in_0_base = b * stride_in_0 + h * (stride_in_0 // W) + w  # simplified for the specific shapes
    
    # Actually, let's compute more carefully
    # in_0 shape: [B, C, H, W] = [1, 2, 256, 32] or [1, 2, 256, 8]
    # We want to compute for each (h, w) position:
    # out[b, h, w] = sum_c(in_0[b, c, h, w] * softmax_c(in_1[b, :, h, w]))
    # Since in_1 has shape [B, C, 1, 1], in_1[b, :, h, w] = in_1[b, :, 0, 0]
    
    # First, compute softmax over the C dimension for this (b, h, w)
    # in_1 is at [B, C, 1, 1], so we need in_1 values at [b, 0, 0, 0] and [b, 1, 0, 0]
    
    # Load the two values for softmax
    in_1_base = b * stride_in_1  # shape [B, C, 1, 1], stride through C
    
    # Compute softmax: exp(x_i) / sum(exp(x_j))
    # For C=2, this is: exp(x0) / (exp(x0) + exp(x1)), exp(x1) / (exp(x0) + exp(x1))
    
    # Load in_1 values
    in_1_0 = tl.load(in_1_ptr + in_1_base + 0 * (stride_in_1 // 2))  # C=0
    in_1_1 = tl.load(in_1_ptr + in_1_base + 1 * (stride_in_1 // 2))  # C=1
    
    # Compute softmax
    max_val = tl.maximum(in_1_0, in_1_1)
    exp_0 = tl.exp(in_1_0 - max_val)
    exp_1 = tl.exp(in_1_1 - max_val)
    sum_exp = exp_0 + exp_1
    softmax_0 = exp_0 / sum_exp
    softmax_1 = exp_1 / sum_exp
    
    # Now compute weighted sum over C for each channel
    # For each output channel c (0 to C-1), compute sum over input channels
    # out[c, h, w] = in_0[c, 0, h, w] * softmax_0 + in_0[c, 1, h, w] * softmax_1
    
    # Compute output for this (h, w) position across all output channels
    # in_0 is [B, C_in, H, W], out is [B, C_out, H, W], where C_in=C_out=256
    # We compute each output channel independently
    
    # Since C=256 is not too large and we have limited parallelism,
    # let's compute all 256 output channels sequentially in the kernel
    # Actually, we should parallelize over C for better performance
    
    # For now, let's do a simpler approach: compute weighted sum for all C
    # Each program computes one (h,w) position, iterating over all C
    result = 0.0
    
    # For channel c, in_0[b, c, h, w] and in_0[b, c+128, h, w] can be loaded
    # We need to load from in_0 which has shape [1, 2, 256, 32, 32]
    # The stride for C dimension is H*W = 1024 or 64
    
    # Let's compute it properly:
    # in_0[b, c, h, w] -> offset = b*stride_in_0 + c*(H*W) + h*W + w
    # Here stride_in_0 = 2*H*W
    
    for c in range(256):
        # Load in_0[b, 0, c, h, w] and in_0[b, 1, c, h, w]
        # in_0 shape: [1, 2, 256, 32, 32] or [1, 2, 256, 8, 8]
        # We need to compute offsets manually
        
        # in_0[b, 0, c, h, w] = b*2*256*H*W + 0*256*H*W + c*H*W + h*W + w
        base_offset = c * H * W + h * W + w
        
        in_0_c0 = tl.load(in_0_ptr + base_offset)  # channel 0
        in_0_c1 = tl.load(in_0_ptr + 256 * H * W + base_offset)  # channel 1
        
        # Weighted sum
        result += in_0_c0 * softmax_0 + in_0_c1 * softmax_1
    
    # Store result
    tl.store(out_ptr + out_offset, result)


@torch.fx.wrap
def fused_softmax_mul_sum(in_0, in_1):
    """
    Fused kernel: sum(in_0 * softmax(in_1, dim=1), dim=1)
    
    in_0: [B, C, H, W] = [1, 2, 256, 32, 32] or [1, 2, 256, 8, 8]
    in_1: [B, C, 1, 1] = [1, 2, 256, 1, 1]
    output: [B, H, W] = [1, 256, 32, 32] or [1, 256, 8, 8]
    """
    B, C_in, C_out, H, W = in_0.shape
    # Note: in_1 has shape [B, C_in, 1, 1]
    
    # Output shape: [B, C_out, H, W] -> [1, 256, 32, 32]
    out_shape = (B, C_out, H, W)
    out = torch.empty(out_shape, device=in_0.device, dtype=in_0.dtype)
    
    # Flatten spatial dimensions for grid
    # Each program processes one (b, h, w) position
    N = B * H * W
    grid = (N,)
    
    # Compute strides
    # in_0: [B, C_in, C_out, H, W] -> stride = [C_in*C_out*H*W, C_out*H*W, H*W, W, 1]
    # But actually let's compute based on the actual tensor
    stride_in_0_0 = in_0.stride(0)
    stride_in_0_1 = in_0.stride(1)
    stride_in_0_2 = in_0.stride(2)
    stride_in_0_3 = in_0.stride(3)
    stride_in_0_4 = in_0.stride(4)
    
    stride_in_1_0 = in_1.stride(0)
    stride_in_1_1 = in_1.stride(1)
    stride_in_1_2 = in_1.stride(2)
    stride_in_1_3 = in_1.stride(3)
    stride_in_1_4 = in_1.stride(4)
    
    # For the kernel, we need:
    # - in_0: [B, C_in, C_out, H, W], we access in_0[b, ci, co, h, w]
    # - in_1: [B, C_in, 1, 1], we access in_1[b, ci, 0, 0]
    # - out: [B, C_out, H, W], we access out[b, co, h, w]
    
    # We need to pass the right strides
    # Let's simplify: use numel and compute offsets inside the kernel
    
    fused_softmax_mul_sum_kernel[grid](
        in_0, in_1, out,
        N, C_in, H, W,
        in_0.numel(), in_1.numel(),
        out.numel(),
        BLOCK_SIZE=1024,
    )
    
    return out


def replacement_func():
    return fused_softmax_mul_sum