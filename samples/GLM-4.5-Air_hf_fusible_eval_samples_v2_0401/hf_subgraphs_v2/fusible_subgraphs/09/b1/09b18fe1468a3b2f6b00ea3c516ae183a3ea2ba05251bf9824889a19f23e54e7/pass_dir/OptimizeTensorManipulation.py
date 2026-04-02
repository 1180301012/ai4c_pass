import torch
import triton
import triton.language as tl

def pattern(in_0, in_4):
    # Tensor manipulation pattern
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10

def replacement_args(in_0, in_4):
    return (in_0, in_4)

@triton.jit
def tensor_manipulation_kernel(
    codewords_ptr, x_ptr, out_ptr,
    B: tl.constexpr, N: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute indices
    batch_idx = tl.program_id(0)
    n_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    k_idx = tl.program_id(2)
    
    # Create masks for bounds checking
    n_mask = n_idx < N
    
    # Load expanded x: [B, N, K, D] 
    x_ptr_idx = batch_idx * B * N * K * D + n_idx[:, None] * K * D + k_idx * D + tl.arange(0, D)[None, :]
    x_loaded = tl.load(x_ptr + x_ptr_idx, mask=n_mask[:, None] and tl.arange(0, D)[None, :] < D, other=0.0)
    
    # Load codewords: [B, 1, K, D] -> broadcast to [B, N, K, D]
    codewords_ptr_idx = batch_idx * B * 1 * K * D + 0 * 1 * K * D + k_idx * D + tl.arange(0, D)[None, :]
    codewords_loaded = tl.load(codewords_ptr + codewords_ptr_idx, mask=tl.arange(0, D)[None, :] < D, other=0.0)
    
    # Expand codewords to match x dimensions (broadcast)
    codewords_bcast = tl.broadcast_to(codewords_loaded, (BLOCK_SIZE, D))
    
    # Element-wise subtraction
    result = x_loaded - codewords_bcast
    
    # Store result
    out_ptr_idx = batch_idx * B * N * K * D + n_idx[:, None] * K * D + k_idx * D + tl.arange(0, D)[None, :]
    tl.store(out_ptr + out_ptr_idx, result, mask=n_mask[:, None] and tl.arange(0, D)[None, :] < D)

@torch.fx.wrap
def optimized_tensor_manipulation(in_0, in_4):
    # Get tensor shapes
    B, N, K, D = 1, 4096, 32, 512  # From the computation
    
    # Create output tensor
    out = torch.zeros((B, N, K, D), dtype=in_4.dtype, device=in_4.device)
    
    # Block size for parallel computation
    BLOCK_SIZE = 256  # Optimize for GPU warps
    
    # Calculate grid dimensions
    num_blocks_N = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    tensor_manipulation_kernel[(B, num_blocks_N, K)](
        in_0.view((1, 1, 32, 512)),  # Reshape codewords
        in_4.unsqueeze(2).expand((1, 4096, 32, 512)),  # Expanded x
        out,
        B, N, K, D,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_tensor_manipulation