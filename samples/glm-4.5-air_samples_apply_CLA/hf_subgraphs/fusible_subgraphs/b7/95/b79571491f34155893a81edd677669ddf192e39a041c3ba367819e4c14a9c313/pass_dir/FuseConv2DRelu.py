import torch
import triton
import triton.language as tl

@torch.fx.wrap
def conv2d_relu_fused(x, y, z):
    """Fused Conv2D + ReLU wrapper using Triton"""
    return triton_conv2d_relu_fused(x, y, z)

@triton.jit
def triton_conv2d_relu_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr,
    N, C_in, H, W, C_out,
    stride_h, stride_w, pad_h, pad_w,
    dilation_h, dilation_w, groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < C_out
    n_mask = n_offsets < (H * W)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, C_in, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < C_in
        
        x_vals = tl.load(
            x_ptr + (
                (n_offsets // W) * (C_in * H * W) + 
                k_offset[:, None] * (H * W) + 
                (n_offsets % W)[None, :]
            ),
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        y_vals = tl.load(
            y_ptr + (
                m_offsets[:, None] * (C_in * 7 * 7) + 
                k_offset[None, :] * (7 * 7) + 
                tl.arange(0, 7 * 7)[None, :]
            ),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, 7, 7)
        
        x_patches = x_vals.reshape(BLOCK_SIZE_K, 7, 7, BLOCK_SIZE_M, BLOCK_SIZE_N)
        x_patches = tl.transpose(x_patches, (3, 0, 1, 2, 4))
        
        conv_result = tl.sum(x_patches * y_vals[:, :, :, :, None], axis=(1, 2, 3))
        accumulator += conv_result
    
    bias_vals = tl.load(
        z_ptr + m_offsets,
        mask=m_mask,
        other=0.0
    )[:, None]
    
    out = tl.maximum(accumulator + bias_vals, 0.0)
    
    tl.store(
        out_ptr + (
            m_offsets[:, None] * (H * W) + 
            n_offsets[None, :]
        ),
        out,
        mask=m_mask[:, None] & n_mask[None, :]
    )

@triton.jit
def triton_conv2d_relu_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr,
    N, C_in, H, W, C_out,
    stride_h, stride_w, pad_h, pad_w,
    dilation_h, dilation_w, groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < C_out
    n_mask = n_offsets < (H * W)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, C_in, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < C_in
        
        x_vals = tl.load(
            x_ptr + (
                (n_offsets // W) * (C_in * H * W) + 
                k_offset[:, None] * (H * W) + 
                (n_offsets % W)[None, :]
            ),
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        y_vals = tl.load(
            y_ptr + (
                m_offsets[:, None] * (C_in * 7 * 7) + 
                k_offset[None, :] * (7 * 7) + 
                tl.arange(0, 7 * 7)[None, :]
            ),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, 7, 7)
        
        x_patches = x_vals.reshape(BLOCK_SIZE_K, 7, 7, BLOCK_SIZE_M, BLOCK_SIZE_N)
        x_patches = tl.transpose(x_patches, (3, 0, 1, 2, 4))
        
        conv_result = tl.sum(x_patches * y_vals[:, :, :, :, None], axis=(1, 2, 3))
        accumulator += conv_result
    
    bias_vals = tl.load(
        z_ptr + m_offsets,
        mask=m_mask,
        other=0.0
    )[:, None]
    
    out = tl.maximum(accumulator + bias_vals, 0.0)
    
    tl.store(
        out_ptr + (
            m_offsets[:, None] * (H * W) + 
            n_offsets[None, :]
        ),
        out,
        mask=m_mask[:, None] & n_mask[None, :]
    )

@triton.jit
def triton_conv2d_relu_kernel_optimized(
    x_ptr, y_ptr, z_ptr, out_ptr,
    N, C_in, H, W, C_out,
    stride_h, stride_w, pad_h, pad_w,
    dilation_h, dilation_w, groups
):
    # Optimized block sizes for our specific tensor shapes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 1024
    BLOCK_SIZE_K = 32
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < C_out
    n_mask = n_offsets < (H * W)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, C_in, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offset < C_in
        
        x_vals = tl.load(
            x_ptr + (
                (n_offsets // W) * (C_in * H * W) + 
                k_offset[:, None] * (H * W) + 
                (n_offsets % W)[None, :]
            ),
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        y_vals = tl.load(
            y_ptr + (
                m_offsets[:, None] * (C_in * 7 * 7) + 
                k_offset[None, :] * (7 * 7) + 
                tl.arange(0, 7 * 7)[None, :]
            ),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K, 7, 7)
        
        x_patches = x_vals.reshape(BLOCK_SIZE_K, 7, 7, BLOCK_SIZE_M, BLOCK_SIZE_N)
        x_patches = tl.transpose(x_patches, (3, 0, 1, 2, 4))
        
        conv_result = tl.sum(x_patches * y_vals[:, :, :, :, None], axis=(1, 2, 3))
        accumulator += conv_result
    
    bias_vals = tl.load(
        z_ptr + m_offsets,
        mask=m_mask,
        other=0.0
    )[:, None]
    
    out = tl.maximum(accumulator + bias_vals, 0.0)
    
    tl.store(
        out_ptr + (
            m_offsets[:, None] * (H * W) + 
            n_offsets[None, :]
        ),
        out,
        mask=m_mask[:, None] & n_mask[None, :]
    )

def triton_conv2d_relu_fused(x, y, z):
    N, C_in, H, W = x.shape
    C_out = y.shape[0]
    
    y = y.cuda()
    z = z.cuda()
    
    out = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
    
    # Optimized grid configuration
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 1024
    grid_m = (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (H * W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    triton_conv2d_relu_kernel_optimized[grid](
        x, y, z, out, 
        N, C_in, H, W, C_out,
        1, 1, 0, 0, 1, 1, 1
    )
    
    return out

def pattern(conv_out, relu_out):
    # Pattern: just the ReLU (we replace the entire conv2d+relu sequence)
    return relu_out

def replacement_args(conv_out, relu_out):
    return (conv_out, relu_out)

def replacement_func():
    return conv2d_relu_fused