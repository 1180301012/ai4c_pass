import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_0):
    # The pattern should match torch.conv2d with the exact parameters used in the graphs
    conv2d = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), 1, groups=1)
    return conv2d

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C_in, H_in, W_in,
    C_out, K_H, K_W,
    stride_H, stride_W,
    pad_H, pad_W,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program ID and total programs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(C_out, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N * H_out * W_out, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(C_in // groups, BLOCK_SIZE_K)
    
    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // (num_pid_m * num_pid_n)
    
    # Compute output dimensions
    H_out = (H_in + 2 * pad_H - K_H) // stride_H + 1
    W_out = (W_in + 2 * pad_W - K_W) // stride_W + 1
    
    # Block ranges
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offset = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Mask for bounds checking
    m_mask = m_offsets < C_out
    n_mask = n_offset < N * H_out * W_out
    k_mask = k_offset < (C_in // groups)
    
    # Initialize output accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(C_in // groups, BLOCK_SIZE_K)):
        # Load weight
        weight_ptrs = weight_ptr + (m_offsets[:, None] * (K_H * K_W * (C_in // groups)) + 
                                   k_offset[None, :] * (K_H * K_W) + 
                                   tl.arange(0, K_H * K_W)[None, :])
        weight = tl.load(weight_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
        weight = weight.reshape((BLOCK_SIZE_M, BLOCK_SIZE_K, K_H, K_W))
        
        # Load input and bias
        input_ptrs = x_ptr + (k_offset[None, :] * H_in * W_in + 
                             n_offset[None, :])
        input_val = tl.load(input_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        
        bias_ptrs = bias_ptr + m_offsets
        bias = tl.load(bias_ptrs, mask=m_mask, other=0.0).to(tl.float32)[:, None]
        
        # Compute convolution
        for kh in range(K_H):
            for kw in range(K_W):
                # Extract patch
                patch = input_val
                # Pad manually or ensure bounds are handled
                if kh > 0:
                    offset = (kh - pad_H) * stride_H
                    patch = tl.where(offset >= 0, patch, 0.0)
                if kw > 0:
                    offset = (kw - pad_W) * stride_W
                    patch = tl.where(offset >= 0, patch, 0.0)
                
                # Accumulate
                accumulator += (weight[:, :, kh, kw] * patch).sum(1)
    
    # Add bias
    accumulator += bias
    
    # Store output
    out_ptrs = out_ptr + (m_offsets[:, None] * (H_out * W_out * (C_out // BLOCK_SIZE_M)) + 
                         n_offset[None, :])
    tl.store(out_ptrs, accumulator.to(tl.float32), mask=(m_mask[:, None] & n_mask[None, :]))

@torch.fx.wrap
def optimized_conv2d(x, weight, bias):
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_H, K_W = weight.shape
    stride_H, stride_W = 1, 1
    pad_H, pad_W = 3, 3
    groups = 1
    
    # Calculate output dimensions
    H_out = (H_in + 2 * pad_H - K_H) // stride_H + 1
    W_out = (W_in + 2 * pad_W - K_W) // stride_W + 1
    
    # Initialize output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Set block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 1024
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_size = (C_out // BLOCK_SIZE_M) * (N * H_out * W_out // BLOCK_SIZE_N) * ((C_in // groups) // BLOCK_SIZE_K)
    
    # Launch kernel
    conv2d_kernel[grid_size](
        x=x,
        weight=weight,
        bias=bias,
        out=out,
        N=N, C_in=C_in, H_in=H_in, W_in=W_in,
        C_out=C_out, K_H=K_H, K_W=K_W,
        stride_H=stride_H, stride_W=stride_W,
        pad_H=pad_H, pad_W=pad_W,
        groups=groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return optimized_conv2d