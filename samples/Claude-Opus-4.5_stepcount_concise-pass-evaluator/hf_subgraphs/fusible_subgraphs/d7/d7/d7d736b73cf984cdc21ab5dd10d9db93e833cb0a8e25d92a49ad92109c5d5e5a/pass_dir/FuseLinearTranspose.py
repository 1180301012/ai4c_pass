import torch
import triton
import triton.language as tl

# Pattern matching function - must match model.py exactly
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for fused linear + transpose
# Computes: output[b, n, m] = sum_k(input[b, m, k] * weight[n, k]) + bias[n]
# This is equivalent to: (weight @ input[b].T) + bias (broadcasted)
@triton.autotune(
    configs=[
        # Best configs for batch 32 (M=768, N=196, K=196)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def fused_linear_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, M, K, N,  # B=batch, M=768, K=196, N=196
    stride_ib, stride_im, stride_ik,  # input strides
    stride_wn, stride_wk,  # weight strides
    stride_ob, stride_on, stride_om,  # output strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B * num_tiles_n * num_tiles_m)
    # Using 1D grid for better occupancy
    
    pid = tl.program_id(0)
    
    # Calculate tile and batch indices
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_tiles_m * num_tiles_n
    
    batch_id = pid // tiles_per_batch
    tile_id = pid % tiles_per_batch
    tile_n = tile_id // num_tiles_m
    tile_m = tile_id % num_tiles_m
    
    # Compute offsets
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    
    # Base pointers for this batch
    input_batch_ptr = input_ptr + batch_id * stride_ib
    output_batch_ptr = output_ptr + batch_id * stride_ob
    
    # Compute: output[b, n, m] = sum_k(input[b, m, k] * weight[n, k]) + bias[n]
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K
        
        # Load weight tile [BLOCK_N, BLOCK_K]
        weight_ptrs = weight_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        weight_mask = (offs_n[:, None] < N) & k_mask[None, :]
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Load input tile [BLOCK_M, BLOCK_K]
        input_ptrs = input_batch_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik
        input_mask = (offs_m[:, None] < M) & k_mask[None, :]
        input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Matmul: acc[BLOCK_N, BLOCK_M] += weight[BLOCK_N, BLOCK_K] @ input[BLOCK_M, BLOCK_K].T
        acc += tl.dot(weight_tile, tl.trans(input_tile))
    
    # Load bias and add
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc = acc + bias_vals[:, None]
    
    # Store output
    output_ptrs = output_batch_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om
    output_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(output_ptrs, acc, mask=output_mask)

@torch.fx.wrap
def fused_linear_transpose(in_0, in_1, in_2):
    """
    Fused linear + transpose operation.
    in_0: bias [N]
    in_1: weight [N, K]
    in_2: input [B, M, K]
    Output: [B, N, M]
    """
    bias = in_0
    weight = in_1
    input_tensor = in_2
    
    B, M, K = input_tensor.shape  # [B, 768, 196]
    N = weight.shape[0]  # 196
    
    # Output shape: [B, N, M] = [B, 196, 768]
    output = torch.empty((B, N, M), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Grid: 1D grid with all tiles flattened
    def grid(META):
        return (B * triton.cdiv(N, META['BLOCK_N']) * triton.cdiv(M, META['BLOCK_M']),)
    
    fused_linear_transpose_kernel[grid](
        input_tensor, weight, bias, output,
        B, M, K, N,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
    )
    
    return output

def replacement_func():
    return fused_linear_transpose