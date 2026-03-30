import torch
import triton
import triton.language as tl

# Pattern matching function for matmul + transpose + contiguous
def pattern(tmp_3, in_3):
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

# Argument extraction function
def replacement_args(tmp_3, in_3):
    return (tmp_3, in_3)

# Optimized Triton kernel for fused matmul + transpose + contiguous
@triton.jit
def fused_matmul_transpose_kernel(
    attention_weights_ptr,
    value_layer_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program processes one MxN block of the output
    m = tl.program_id(0)
    n = tl.program_id(1)
    b = tl.program_id(2)
    
    # Create accumulator for the block
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, T, TILE_K):
        for i in range(BLOCK_M):
            for j in range(BLOCK_N):
                if m * BLOCK_M + i < T and n * BLOCK_N + j < D and k + 0 < T:
                    # Load attention weights: [B, H, T, T] -> read [i, k] from position m
                    w_offset = b * H * T * T + (m * BLOCK_M + i) * T + k
                    weight = tl.load(attention_weights_ptr + w_offset, mask=(m * BLOCK_M + i < T and k < T))
                    
                    # Load value layer: [B, H, T, D] -> read [k, j] from position n
                    v_offset = b * H * T * D + k * D + (n * BLOCK_N + j)
                    value = tl.load(value_layer_ptr + v_offset, mask=(k < T and n * BLOCK_N + j < D))
                    
                    # Multiply and add to accumulator
                    accumulator[i, j] += weight * value
    
    # Store transposed result: output should be [B, D, T, H]
    # Our accumulator is for [T, D] block, but we need [D, T] in final output
    for i in range(BLOCK_M):
        for j in range(BLOCK_N):
            if m * BLOCK_M + i < T and n * BLOCK_N + j < D:
                # Store in transposed position: [D, T] instead of [T, D]
                output_row = n * BLOCK_N + j  # corresponds to D dimension
                output_col = m * BLOCK_M + i  # corresponds to T dimension
                output_offset = b * D * T * H + output_row * T * H + output_col * H
                tl.store(output_ptr + output_offset, accumulator[i, j])

@torch.fx.wrap  
def fused_matmul_transpose_contiguous(tmp_3, in_3):
    B, H, T, T_orig = tmp_3.shape
    B_v, H_v, T_v, D = in_3.shape
    
    # Verify shapes are compatible
    assert B == B_v and H == H_v and T == T_v, "Shape mismatch in attention computation"
    
    # The output should be [B, D, T, H] after transpose(0,2,1,3)
    output = torch.empty((B, D, T, H), dtype=attention_weights.dtype, device=attention_weights.device)
    
    # Tile sizes for matmul
    BLOCK_M = 32  # Tile size for T dimension
    BLOCK_N = 32  # Tile size for D dimension  
    TILE_K = 32   # Tile size for K dimension
    
    # Calculate grid dimensions
    M_blocks = (T + BLOCK_M - 1) // BLOCK_M
    N_blocks = (D + BLOCK_N - 1) // BLOCK_N
    
    # Launch kernel
    fused_matmul_transpose_kernel[(M_blocks, N_blocks, B,)](
        attention_weights_ptr=tmp_3,
        value_layer_ptr=in_3,
        output_ptr=output,
        B=B,
        H=H,
        T=T,
        D=D,
        TILE_K=TILE_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_matmul_transpose_contiguous