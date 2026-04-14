import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for Linear + Sigmoid + View + Element-wise multiplication
    This matches the computation pattern found in all target graphs
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    # Match the exact view operations used in different graphs
    batch_size = in_2.shape[0]
    tmp_4 = tmp_3.view(batch_size, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def linear_sigmoid_view_mul_kernel(
    bias_ptr,           # [64]
    weight_ptr,         # [64, 8] 
    input_ptr,          # [N, 8]
    multiply_ptr,       # [N, 64, H, W]
    output_ptr,         # [N, 64, H, W]
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: Linear + Sigmoid + View + Element-wise multiplication
    
    This kernel fuses the entire computation pipeline:
    1. Linear operation: output = input @ weight.T + bias
    2. Sigmoid activation: output = sigmoid(output)  
    3. Reshape to [N, 64, 1, 1] implicitly in the kernel
    4. Element-wise multiplication with multiply_ptr
    """
    # Get program ID and offsets
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # Output feature dimension (0-63)
    
    # Create offsets for matrix multiplication
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Bounds checking
    m_mask = m_offsets < N
    n_mask = n_offsets < 64
    k_mask = k_offsets < 8
    
    # Load bias [64]
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    
    # Compute linear operation using MMA
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (input features)
    for k in range(0, 8, BLOCK_SIZE_K):
        k_offsets_blocked = k + k_offsets
        k_mask_blocked = k_offsets_blocked < 8
        
        # Load weight chunk [BLOCK_SIZE_N, BLOCK_SIZE_K]  i.e., [BLOCK_SIZE_N, BLOCK_SIZE_K]
        weight = tl.load(
            weight_ptr + n_offsets[:, None] * 8 + k_offsets_blocked[None, :],
            mask=n_mask[:, None] & k_mask_blocked[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Load input chunk [BLOCK_SIZE_M, BLOCK_SIZE_K] i.e., [BLOCK_SIZE_M, BLOCK_SIZE_K]
        input_data = tl.load(
            input_ptr + m_offsets[:, None] * 8 + k_offsets_blocked[None, :],
            mask=m_mask[:, None] & k_mask_blocked[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Matrix multiply accumulate
        acc += tl.dot(input_data, weight, out_layout="MM")
    
    # Add bias and apply sigmoid
    output_linear = acc + bias[None, :]
    output_sigmoid = 1.0 / (1.0 + tl.exp(-output_linear))
    
    # Broadcast to spatial dimensions and multiply with input tensor
    # Output shape: [BLOCK_SIZE_M, BLOCK_SIZE_N, 1, 1]
    output_sigmoid_4d = output_sigmoid[:, :, None, None]
    
    # Load the multiplying tensor [BLOCK_SIZE_M, BLOCK_SIZE_N, H, W]
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
    
    # Element-wise multiplication
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
def linear_sigmoid_view_mul_fusion(in_0, in_1, in_2, in_3):
    """
    Wrapper function for the fused Linear + Sigmoid + View + Multiplication kernel
    """
    N = in_2.shape[0]
    H, W = in_3.shape[2], in_3.shape[3]
    
    # Input shapes validation
    assert in_0.shape == (64,), f"Bias shape mismatch: expected (64,), got {in_0.shape}"
    assert in_1.shape == (64, 8), f"Weight shape mismatch: expected (64, 8), got {in_1.shape}"
    assert in_2.shape == (N, 8), f"Input shape mismatch: expected ({N}, 8), got {in_2.shape}"
    assert in_3.shape == (N, 64, H, W), f"Multiply tensor shape mismatch: expected ({N}, 64, {H}, {W}), got {in_3.shape}"
    
    # Output tensor
    output = torch.empty((N, 64, H, W), dtype=in_3.dtype, device=in_3.device)
    
    # Block sizes for matrix multiplication (optimized for small matrices)
    BLOCK_SIZE_M = 64    # Process multiple batches together
    BLOCK_SIZE_N = 64    # Process all output features together
    BLOCK_SIZE_K = 8     # Process all input features together (full K dimension)
    
    # Calculate grid sizes
    grid_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (64 + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    linear_sigmoid_view_mul_kernel[grid_m, grid_n](
        in_0,    # bias_ptr
        in_1,    # weight_ptr  
        in_2,    # input_ptr
        in_3,    # multiply_ptr
        output,  # output_ptr
        N,       # N (batch size)
        H,       # Height dimension
        W,       # Width dimension
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return linear_sigmoid_view_mul_fusion