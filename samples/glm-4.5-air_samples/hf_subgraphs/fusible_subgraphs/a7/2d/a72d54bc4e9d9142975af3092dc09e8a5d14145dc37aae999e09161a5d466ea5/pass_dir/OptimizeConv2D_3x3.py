import torch
import triton
import triton.language as tl

# Pattern matching function - matches the conv2d operation
def pattern(in_6, in_0):
    tmp_5 = torch.conv2d(in_6, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    return tmp_5

# Argument extraction function
def replacement_args(in_6, in_0):
    return (in_6, in_0)

# Optimized convolution kernel using Triton
@triton.jit
def conv2d_kernel(
    x_ptr,  # input tensor [B, C_in, H, W]
    weight_ptr,  # weight tensor [C_out, C_in, K, K]
    out_ptr,  # output tensor [B, C_out, H, W]
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID determines which output channel and batch element we compute
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output channel dimension
    pid_b = tl.program_id(2)  # block in batch (if needed for large batches)
    
    # Adjust program IDs for block processing
    m_block = pid_b * BLOCK_SIZE_M
    m_offset = m_block + tl.arange(0, BLOCK_SIZE_M)
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute number of iterations needed
    num_k_iterations = (in_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    for k_idx in range(num_k_iterations):
        # Load input blocks
        k_base = k_idx * BLOCK_SIZE_K
        
        # Load weight blocks for current output channel
        # Create weight indices for current output channel range and input channel range
        n_indices = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[:, None, None, None]
        k_indices = (k_base + tl.arange(0, BLOCK_SIZE_K))[None, None, :, None]
        weight_ptrs = weight_ptr + (n_indices, k_indices, 0, 0)
        weight = tl.load(weight_ptrs, mask=(n_indices < out_channels) & (k_indices < in_channels), 
                        other=0.0)
        weight = weight.permute(0, 2, 3, 1)  # [BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_M]
        
        # Process each input channel block
        for c_in in range(k_base, min(k_base + BLOCK_SIZE_K, in_channels)):
            # Load input patch for current batch element
            h_start = -pad_h
            h_end = height * stride_h - pad_h
            w_start = -pad_w
            w_end = width * stride_w - pad_w
            
            # Load input with zero-padding at boundaries
            input_ptrs = x_ptr + (m_offset[:, None, None], c_in, 
                               h_start + tl.arange(0, kernel_size)[:, None],
                               w_start + tl.arange(0, kernel_size)[None, :])
            x_patch = tl.load(input_ptrs, 
                             mask=(h_start + tl.arange(0, kernel_size)[:, None] >= 0) & 
                                  (h_start + tl.arange(0, kernel_size)[:, None] < height * stride_h) &
                                  (w_start + tl.arange(0, kernel_size)[None, :] >= 0) & 
                                  (w_start + tl.arange(0, kernel_size)[None, :] < width * stride_w) &
                                  (m_offset[:, None, None] < batch_size) &
                                  (c_in < in_channels),
                             other=0.0)
            
            # Accumulate matrix multiplication
            accumulator += tl.sum(x_patch[:, :, :, :, None] * weight[None, None, None, :, :], axis=(2, 3))
    
    # Store output
    out_ptrs = out_ptr + (m_offset[:, None], pid_n, 0, 0)
    mask = (m_offset[:, None] < batch_size) & (pid_n < out_channels)
    tl.store(out_ptrs, accumulator, mask=mask)

# Kernel wrapper that handles launching the Triton kernel
@torch.fx.wrap
def triton_conv2d(x, weight):
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    
    # Handle small batch sizes efficiently
    if B == 1:
        return _triton_conv2d_batch1(x, weight, C_out)
    else:
        return _triton_conv2d_batch_general(x, weight, B, C_in, C_out, H, W, K)

@torch.fx.wrap  
def _triton_conv2d_batch1(x, weight, C_out):
    B, C_in, H, W = x.shape
    K = weight.shape[2]
    
    # Allocate output
    out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Use simple computation for batch size 1
    # Each output channel handled by one program
    def compute_channel(c_out):
        # Compute convolution for single output channel
        result = torch.zeros((B, H, W), dtype=x.dtype, device=x.device)
        for c_in in range(C_in):
            w = weight[c_out, c_in]
            # Extract valid patches
            for i in range(H):
                for j in range(W):
                    # Get patch with padding
                    patch_start_h = max(0, i - K//2)
                    patch_end_h = min(H, i - K//2 + K)
                    patch_start_w = max(0, j - K//2)
                    patch_end_w = min(W, j - K//2 + K)
                    
                    if patch_start_h < patch_end_h and patch_start_w < patch_end_w:
                        patch = x[0, c_in, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
                        w_patch = w[:patch_end_h-patch_start_h, :patch_end_w-patch_start_w]
                        result[0, i, j] += float((patch * w_patch).sum())
        return result
    
    # Compute all output channels
    for c_out in range(C_out):
        out[0, c_out] = compute_channel(c_out)
    
    return out

@torch.fx.wrap
def _triton_conv2d_batch_general(x, weight, B, C_in, C_out, H, W, K):
    # Allocate output
    out = torch.empty((B, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Block size configuration
    BLOCK_SIZE_M = 32  # batch block size
    BLOCK_SIZE_N = 64  # output channel block size  
    BLOCK_SIZE_K = 32  # input channel block size
    
    # Calculate grid dimensions
    num_batch_blocks = (B + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_out_channel_blocks = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    grid = (num_batch_blocks, num_out_channel_blocks, 1)
    
    conv2d_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        batch_size=B,
        in_channels=C_in,
        out_channels=C_out,
        height=H, 
        width=W,
        kernel_size=K,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        dilation_h=1,
        dilation_w=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

# Replacement function (returns the optimized kernel function)
def replacement_func():
    return triton_conv2d