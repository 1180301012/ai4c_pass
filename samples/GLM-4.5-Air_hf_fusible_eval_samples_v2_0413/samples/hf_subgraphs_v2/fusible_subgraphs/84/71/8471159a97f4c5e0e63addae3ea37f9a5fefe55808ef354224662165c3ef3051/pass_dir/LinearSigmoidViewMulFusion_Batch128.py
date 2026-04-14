import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for Linear + Sigmoid + View + Element-wise multiplication
    Specifically for graphs with batch size 128
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(128, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def linear_sigmoid_view_mul_kernel_b128(
    bias_ptr,           # [64]
    weight_ptr,         # [64, 8] 
    input_ptr,          # [128, 8]
    multiply_ptr,       # [128, 64, H, W]
    output_ptr,         # [128, 64, H, W]
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for Linear + Sigmoid + View + Element-wise multiplication
    Optimized for batch size = 128
    """
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # Output feature dimension (0-63)
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Bounds checking
    m_mask = m_offsets < 128
    n_mask = n_offsets < 64
    k_mask = k_offsets < 8
    
    # Load bias [64]
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Compute linear operation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, 8, BLOCK_SIZE_K):
        k_offsets_blocked = k + k_offsets
        k_mask_blocked = k_offsets_blocked < 8
        
        weight = tl.load(
            weight_ptr + n_offsets[:, None] * 8 + k_offsets_blocked[None, :],
            mask=n_mask[:, None] & k_mask_blocked[None, :],
            other=0.0
        ).to(tl.float32)
        
        input_data = tl.load(
            input_ptr + m_offsets[:, None] * 8 + k_offsets_blocked[None, :],
            mask=m_mask[:, None] & k_mask_blocked[None, :],
            other=0.0
        ).to(tl.float32)
        
        acc += tl.dot(input_data, weight)
    
    # Add bias and apply sigmoid
    output_linear = acc + bias[None, :]
    output_sigmoid = 1.0 / (1.0 + tl.exp(-output_linear))
    
    # Broadcast and multiply
    output_sigmoid_4d = output_sigmoid[:, :, None, None]
    
    # Load multiply tensor and do element-wise multiplication
    multiply_data = tl.load(
        multiply_ptr + m_offsets[:, None, None, None] * (64 * H * W) + 
        n_offsets[None, :, None, None] * (H * W) +
        tl.arange(0, H)[None, None, :, None] * W +
        tl.arange(0, W)[None, None, None, :],
        mask=m_mask[:, None, None, None] & 
             n_mask[None, :, None, None] &
             (tl.arange(0, H)[None, None, :, None] < H) &
             (tl.arange(0, W)[None, None, None, :] < W),
        other=0.0
    )
    
    final_output = output_sigmoid_4d * multiply_data.to(output_sigmoid.dtype)
    
    # Store result
    tl.store(
        output_ptr + m_offsets[:, None, None, None] * (64 * H * W) + 
        n_offsets[None, :, None, None] * (H * W) +
        tl.arange(0, H)[None, None, :, None] * W +
        tl.arange(0, W)[None, None, None, :],
        final_output,
        mask=m_mask[:, None, None, None] & 
             n_mask[None, :, None, None] &
             (tl.arange(0, H)[None, None, :, None] < H) &
             (tl.arange(0, W)[None, None, None, :] < W)
    )

@torch.fx.wrap
def linear_sigmoid_view_mul_fusion_b128(in_0, in_1, in_2, in_3):
    """
    Wrapper for batch size 128
    """
    assert in_2.shape[0] == 128, f"Expected batch size 128, got {in_2.shape[0]}"
    H, W = in_3.shape[2], in_3.shape[3]
    
    output = torch.empty((128, 64, H, W), dtype=in_3.dtype, device=in_3.device)
    
    BLOCK_SIZE_M = 32  # Process smaller chunks for better occupancy
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 8
    
    grid_m = (128 + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (64 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    linear_sigmoid_view_mul_kernel_b128[grid_m, grid_n](
        in_0, in_1, in_2, in_3, output, H, W, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return linear_sigmoid_view_mul_fusion_b128