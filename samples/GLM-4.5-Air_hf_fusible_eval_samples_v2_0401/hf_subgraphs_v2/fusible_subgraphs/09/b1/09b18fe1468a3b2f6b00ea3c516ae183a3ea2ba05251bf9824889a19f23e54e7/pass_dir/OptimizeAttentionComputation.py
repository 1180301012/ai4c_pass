import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Simple pattern: view operation
    tmp_6 = in_0.view((1, 1, 32, 512))
    return tmp_6

def replacement_args(in_0):
    return (in_0,)

@torch.jit.script
def optimized_view(in_0):
    # Simple optimized view operation
    return in_0.view((1, 1, 32, 512))

def replacement_func():
    return optimized_view

@triton.jit
def attention_kernel(
    x1_ptr, x2_ptr, scale_ptr, 
    out_ptr,
    B: tl.constexpr, N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute batch and head indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    n_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for bounds checking
    n_mask = n_idx < N
    
    # Load batch and head specific data
    stride_B = B * N * K
    stride_N = N * K
    stride_K = K
    
    # Load x1 slice: [B, N, K, D] -> for current batch and head
    x1_slice = tl.load(x1_ptr + batch_idx * stride_B + head_idx * K * N + n_idx[:, None] * stride_K + tl.arange(0, D)[None, :], 
                      mask=n_mask[:, None] and tl.arange(0, D)[None, :] < D, 
                      other=0.0)
    
    # Load x2 slice: [B, 1, K, D] -> broadcast to [B, N, K, D]
    x2_slice = tl.load(x2_ptr + batch_idx * stride_B + 0 * K * N + n_idx[:, None] * stride_K + tl.arange(0, D)[None, :], 
                      mask=n_mask[:, None] and tl.arange(0, D)[None, :] < D, 
                      other=0.0)
    
    # Compute element-wise difference
    diff = x1_slice - x2_slice
    
    # Square and sum along dimension D
    squared = diff * diff
    summed = tl.sum(squared, axis=1)
    
    # Load scale and multiply
    scale_val = tl.load(scale_ptr + batch_idx * stride_B + head_idx * K + tl.arange(0, K)[None, :], 
                       mask=tl.arange(0, K)[None, :] < K, 
                       other=0.0)
    
    weighted = summed * scale_val
    
    # Apply softmax along dimension 2 (K dimension)
    # For softmax, we need to handle numerical stability
    max_val = tl.max(weighted, axis=1)
    max_val = tl.broadcast_to(max_val, (BLOCK_SIZE, K))
    exp_weighted = tl.exp(weighted - max_val)
    sum_exp = tl.sum(exp_weighted, axis=1)
    sum_exp = tl.broadcast_to(sum_exp, (BLOCK_SIZE, K))
    softmax_out = exp_weighted / sum_exp
    
    # Store result
    out_ptr_idx = batch_idx * stride_B + head_idx * K * N + n_idx[:, None] * stride_K + tl.arange(0, K)[None, :]
    tl.store(out_ptr + out_ptr_idx, softmax_out, mask=n_mask[:, None] and tl.arange(0, K)[None, :] < K)

@triton.jit
def attention_with_unsqueeze_kernel(
    x1_ptr, x2_ptr, scale_ptr, 
    attention_out_ptr, unsqueezed_out_ptr,
    B: tl.constexpr, N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute batch and head indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    n_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for bounds checking
    n_mask = n_idx < N
    
    # Load batch and head specific data
    stride_B = B * N * K
    stride_N = N * K
    stride_K = K
    
    # Load x1 slice: [B, N, K, D] -> for current batch and head
    x1_slice = tl.load(x1_ptr + batch_idx * stride_B + head_idx * K * N + n_idx[:, None] * stride_K + tl.arange(0, D)[None, :], 
                      mask=n_mask[:, None] and tl.arange(0, D)[None, :] < D, 
                      other=0.0)
    
    # Load x2 slice: [B, 1, K, D] -> broadcast to [B, N, K, D]
    x2_slice = tl.load(x2_ptr + batch_idx * stride_B + 0 * K * N + n_idx[:, None] * stride_K + tl.arange(0, D)[None, :], 
                      mask=n_mask[:, None] and tl.arange(0, D)[None, :] < D, 
                      other=0.0)
    
    # Compute element-wise difference
    diff = x1_slice - x2_slice
    
    # Square and sum along dimension D
    squared = diff * diff
    summed = tl.sum(squared, axis=1)
    
    # Load scale and multiply
    scale_val = tl.load(scale_ptr + batch_idx * stride_B + head_idx * K + tl.arange(0, K)[None, :], 
                       mask=tl.arange(0, K)[None, :] < K, 
                       other=0.0)
    
    weighted = summed * scale_val
    
    # Apply softmax along dimension 2 (K dimension)
    max_val = tl.max(weighted, axis=1)
    max_val = tl.broadcast_to(max_val, (BLOCK_SIZE, K))
    exp_weighted = tl.exp(weighted - max_val)
    sum_exp = tl.sum(exp_weighted, axis=1)
    sum_exp = tl.broadcast_to(sum_exp, (BLOCK_SIZE, K))
    softmax_out = exp_weighted / sum_exp
    
    # Store attention weights
    attention_out_idx = batch_idx * stride_B + head_idx * K * N + n_idx[:, None] * stride_K + tl.arange(0, K)[None, :]
    tl.store(attention_out_ptr + attention_out_idx, softmax_out, mask=n_mask[:, None] and tl.arange(0, K)[None, :] < K)
    
    # Store unsqueezed attention weights (add dimension 3)
    unsqueezed_out_idx = batch_idx * stride_B + head_idx * K * N + n_idx[:, None] * stride_K * 1 + 0 + tl.arange(0, K)[None, :] * 1
    tl.store(unsqueezed_out_ptr + unsqueezed_out_idx, softmax_out, mask=n_mask[:, None] and tl.arange(0, K)[None, :] < K)

@torch.fx.wrap
def optimized_attention_computation(in_1, in_2, in_3):
    # Get tensor shapes
    B, N, K, D = in_1.shape  # [1, 4096, 32, 512]
    
    # Create output tensor
    out = torch.zeros((B, N, K), dtype=in_1.dtype, device=in_1.device)  # [1, 4096, 32]
    
    # Block size for parallel computation
    BLOCK_SIZE = 256  # Optimize for GPU warps
    
    # Calculate grid dimensions
    num_blocks_N = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    attention_kernel[(B, K, num_blocks_N)](
        in_1,
        in_2, 
        in_3,
        out,
        B, N, K, D,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_attention_computation