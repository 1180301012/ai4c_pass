import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    conv_out = torch.conv2d(x, weight, bias, stride=(1, 1), padding=(0, 0))
    flat_out = conv_out.flatten(2)
    trans_out = flat_out.transpose(1, 2)
    return trans_out

def replacement_args(x, weight, bias, normalize_shape, normalized_shape, eps, cls_token):
    return (x, weight, bias)

@triton.jit
def fused_conv_flatten_transpose_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_in, H_in, W_in, C_out, 
    kernel_H, kernel_W, stride_H, stride_W,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    M = (N * H_in // stride_H * W_in // stride_W)
    K = C_in
    N_out = C_out
    
    grid_y = pid // M
    grid_x = pid % M
    
    for k in range(0, K, BLOCK_SIZE_K):
        for m in range(0, N_out, BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
            
            for n in range(grid_x, M, grid_x + 1):
                h_out = n % (H_in // stride_H)
                w_out = n // (H_in // stride_H)
                
                for k_offset in range(k, min(k + BLOCK_SIZE_K, K)):
                    h_start = h_out * stride_H
                    w_start = w_out * stride_W
                    
                    for kh in range(kernel_H):
                        for kw in range(kernel_W):
                            h_in = h_start + kh
                            w_in = w_start + kw                            
                            if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                x_offset = h_in * W_in + w_in
                                weight_offset = kh * kernel_W + kw
                                x_val = tl.load(x_ptr + n * C_in * H_in * W_in + k_offset * H_in * W_in + x_offset)
                                weight_val = tl.load(weight_ptr + m * C_in * kernel_H * kernel_W + k_offset * kernel_H * kernel_W + weight_offset)
                                acc[m % BLOCK_SIZE_M, k_offset - k] += x_val * weight_val
                
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + m)
                acc += bias_val
            
            if m < N_out and k < K:
                for mm in range(BLOCK_SIZE_M):
                    for kk in range(BLOCK_SIZE_K):
                        if m + mm < N_out and k + kk < K:
                            output_offset = (grid_y * M + grid_x) * N_out * (H_in // stride_H * W_in // stride_H) + (m + mm) * (H_in // stride_H * W_in // stride_H) + n
                            trans_offset = (grid_y * M + grid_x) * (H_in // stride_H * W_in // stride_H) * N_out + n * N_out + (m + mm)
                            if output_offset < M * N_out * (H_in // stride_H * W_in // stride_H):
                                tl.store(out_ptr + trans_offset, acc[mm, kk])

@torch.fx.wrap
def fused_conv_flatten_transpose(x, weight, bias):
    N, C_in, H_in, W_in = x.shape
    C_out, _, kernel_H, kernel_W = weight.shape
    stride_H, stride_W = 1, 1
    
    H_out = (H_in - kernel_H) // stride_H + 1
    W_out = (W_in - kernel_W) // stride_W + 1
    M = N * H_out * W_out
    
    out = torch.empty((N, H_out * W_out, C_out), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 32
    
    grid_size = (M * C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    fused_conv_flatten_transpose_kernel[grid_size,](
        x, weight, bias, out,
        N, C_in, H_in, W_in, C_out,
        kernel_H, kernel_W, stride_H, stride_W,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out

def replacement_func():
    return fused_conv_flatten_transpose