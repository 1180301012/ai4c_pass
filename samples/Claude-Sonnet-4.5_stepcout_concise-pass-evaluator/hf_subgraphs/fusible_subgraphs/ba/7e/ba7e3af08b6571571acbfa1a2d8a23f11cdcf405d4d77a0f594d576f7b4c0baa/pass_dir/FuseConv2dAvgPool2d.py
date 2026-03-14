import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern: 1x1 Conv2D followed by 2x2 AvgPool2D
    in_0: weight tensor [out_channels, in_channels, 1, 1]
    in_1: input tensor [batch, in_channels, height, width]
    """
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_avgpool_matmul_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik, stride_ih, stride_iw,
    stride_wm, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized matmul-style kernel for fused conv1x1 + avgpool2d
    Treats the problem as: output[M, N] = avg(weight[M, K] @ input[K, 4*N])
    where N is the number of output spatial positions
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension (input channels)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K
        
        # Load weights [BLOCK_M, BLOCK_K]
        w_ptrs = weight_ptr + offs_m[:, None] * stride_wm + k_offs[None, :] * stride_wk
        w_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # For avgpool, we need to load 4 input positions and average
        # This is the key optimization: we do 4 matmuls then average
        for pool_idx in range(4):
            # Compute which input position this corresponds to (2x2 pooling)
            dh = pool_idx // 2
            dw = pool_idx % 2
            
            # Load input [BLOCK_K, BLOCK_N] with pooling offset
            i_ptrs = (input_ptr + 
                     k_offs[:, None] * stride_ik +
                     (offs_n[None, :] * 2 + dh) * stride_ih +
                     (dw) * stride_iw)
            i_mask = (k_mask[:, None]) & (offs_n[None, :] < N)
            i = tl.load(i_ptrs, mask=i_mask, other=0.0)
            
            # Accumulate matmul
            acc += tl.dot(w, i)
    
    # Average over 4 pool positions
    acc = acc * 0.25
    
    # Store output
    o_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_avgpool_matmul_kernel_batched(
    input_ptr, weight_ptr, output_ptr,
    batch_size, M, N, K,
    in_height, in_width, out_height, out_width,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized batched kernel with better spatial decoding
    """
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Decode output spatial positions once
    out_h = offs_n // out_width
    out_w = offs_n % out_width
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Base pointers for this batch
    input_base = input_ptr + pid_batch * K * in_height * in_width
    output_base = output_ptr + pid_batch * M * N
    
    # Loop over K dimension (input channels)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K
        
        # Load weights [BLOCK_M, BLOCK_K] - shared across batches
        w_ptrs = weight_ptr + offs_m[:, None] * K + k_offs[None, :]
        w_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # For avgpool, process 4 pooling positions
        # Unroll pooling loop for better performance
        for dh in range(2):
            for dw in range(2):
                # Input positions
                in_h = out_h * 2 + dh
                in_w = out_w * 2 + dw
                
                # Load input [BLOCK_K, BLOCK_N]
                i_ptrs = (input_base + 
                         k_offs[:, None] * in_height * in_width +
                         in_h[None, :] * in_width +
                         in_w[None, :])
                i_mask = (k_mask[:, None]) & (offs_n[None, :] < N)
                i = tl.load(i_ptrs, mask=i_mask, other=0.0)
                
                # Accumulate matmul
                acc += tl.dot(w, i)
    
    # Average over 4 pool positions
    acc = acc * 0.25
    
    # Store output
    o_ptrs = output_base + offs_m[:, None] * N + offs_n[None, :]
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


@torch.fx.wrap
def fused_conv1x1_avgpool2d(input_tensor, weight):
    """
    Fused 1x1 Conv2D + 2x2 AvgPool2D using optimized matmul kernel
    """
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Output dimensions
    out_height = in_height // 2
    out_width = in_width // 2
    
    # Allocate output
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        device=input_tensor.device,
        dtype=input_tensor.dtype
    )
    
    # Reshape for matmul view
    M = out_channels
    N = out_height * out_width
    K = in_channels
    
    # Launch kernel
    def cdiv(a, b):
        return (a + b - 1) // b
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (batch_size, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    
    weight_2d = weight.squeeze(-1).squeeze(-1)
    
    # Launch for all batches at once
    fused_conv1x1_avgpool_matmul_kernel_batched[grid](
        input_tensor, weight_2d, output,
        batch_size, M, N, K,
        in_height, in_width, out_height, out_width,
    )
    
    return output


def replacement_func():
    return fused_conv1x1_avgpool2d