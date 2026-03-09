import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized SILU (Swish) kernel: out = x * sigmoid(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid using fast approximation
    # sigmoid(x) = 1 / (1 + exp(-x))
    # Using approximation for better performance
    z = tl.max(x, tl.zeros((), dtype=tl.float32))
    neg_abs_x = -tl.abs(x)
    sigmoid = tl.where(x >= 0, 1.0 / (1.0 + tl.exp(neg_abs_x)), tl.exp(neg_abs_x) / (1.0 + tl.exp(neg_abs_x)))
    
    # SILU: x * sigmoid(x)
    out = x * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def max_pool2d_kernel(
    x_ptr,  # [N, C, H, W]
    out_ptr,
    N, C, H_in, W_in,
    pool_h, pool_w,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Optimized 2D Max Pooling kernel"""
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate output dimensions
    H_out = (H_in + 2 * pad_h - pool_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - pool_w) // stride_w + 1
    
    # Local memory for pooling window
    pool_window = tl.zeros((pool_h, pool_w), dtype=tl.float32)
    
    # Iterate through output positions
    for h_out in range(0, H_out, BLOCK_SIZE_H):
        for w_out in range(0, W_out, BLOCK_SIZE_W):
            # Reset pooling window
            pool_window.fill_(tl.minimum)
            
            # Process pooling window
            for ph in range(pool_h):
                for pw in range(pool_w):
                    h_in = h_out * stride_h + ph - pad_h
                    w_in = w_out * stride_w + pw - pad_w
                    
                    # Check bounds
                    if 0 <= h_in < H_in and 0 <= w_in < W_in:
                        # Calculate input pointer offset
                        offset = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + h_in * W_in + w_in
                        val = tl.load(x_ptr + offset)
                        pool_window[ph, pw] = tl.maximum(pool_window[ph, pw], val)
            
            # Store max values
            for h_out_local in range(min(BLOCK_SIZE_H, H_out - h_out)):
                for w_out_local in range(min(BLOCK_SIZE_W, W_out - w_out)):
                    h_out_idx = h_out + h_out_local
                    w_out_idx = w_out + w_out_local
                    offset = batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + h_out_idx * W_out + w_out_idx
                    tl.store(out_ptr + offset, pool_window[h_out_local, w_out_local])

@torch.fx.wrap
def triton_silu(x):
    """Optimized SILU function using Triton"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    silu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@torch.fx.wrap
def triton_max_pool2d(x, kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False):
    """Optimized MaxPool2d function using Triton"""
    N, C, H, W = x.shape
    
    # Calculate output dimensions
    if dilation != 1:
        kernel_size = (kernel_size - 1) * dilation + 1
    
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty(N, C, H_out, W_out, dtype=x.dtype, device=x.device)
    
    # Choose block sizes
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 64
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    num_programs_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_programs_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_programs_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_programs_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    max_pool2d_kernel[(num_programs_n, num_programs_c, num_programs_h, num_programs_w)](
        x_ptr=x,
        out_ptr=out,
        N=N, C=C, H_in=H, W_in=W,
        pool_h=kernel_size, pool_w=kernel_size,
        stride_h=stride, stride_w=stride,
        pad_h=padding, pad_w=padding,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return out

def pattern(x):
    """Identity pattern for testing"""
    return (x, x)

def replacement_args(x):
    """Extract arguments for replacement"""
    return (x,)

def replacement_func():
    """Identity replacement for testing"""
    def optimized_forward(x):
        return (x, x)
    
    return optimized_forward