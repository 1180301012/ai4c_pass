import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for Linear + Sigmoid + View + Element-wise multiplication
    Specifically for graphs with batch size 1
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(1, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def linear_sigmoid_view_mul_kernel_b1(
    bias_ptr,           # [64]
    weight_ptr,         # [64, 8] 
    input_ptr,          # [1, 8]
    multiply_ptr,       # [1, 64, H, W]
    output_ptr,         # [1, 64, H, W]
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for Linear + Sigmoid + View + Element-wise multiplication
    Optimized for batch size = 1
    """
    # Get program ID for output features
    pid = tl.program_id(0)  # Output feature dimension (0-63)
    
    # Compute offsets for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 64
    
    # Load bias for this output feature
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Load entire input [8]
    input_data = tl.load(
        input_ptr + tl.arange(0, 8),
        mask=tl.arange(0, 8) < 8,
        other=0.0
    ).to(tl.float32)
    
    # Load entire weight row for this output feature [8]
    weights = tl.load(
        weight_ptr + offsets[:, None] * 8 + tl.arange(0, 8)[None, :],
        mask=mask[:, None] & (tl.arange(0, 8)[None, :] < 8),
        other=0.0
    ).to(tl.float32)
    
    # Compute linear: output = input @ weights.T + bias
    # Use simple element-wise computation for small matrices
    output_linear = bias  # Start with bias
    
    # Manual matrix multiplication: for each output feature, sum over input features
    for k in range(8):
        input_val = input_data[k]
        # Load weight for this input feature across all output features [BLOCK_SIZE]
        weight_col = tl.load(
            weight_ptr + offsets * 8 + k,
            mask=mask,
            other=0.0
        ).to(tl.float32)
        # Add contribution: input_val * weight_col
        output_linear += input_val * weight_col
    
    # Apply sigmoid
    output_sigmoid = 1.0 / (1.0 + tl.exp(-output_linear))
    
    # Create 4D shape [1, 1, 1, 1] and broadcast to [1, 64, H, W]
    output_sigmoid_4d = output_sigmoid
    
    # Load multiply tensor and multiply
    multiply_data = tl.load(
        multiply_ptr + offsets[:, None, None] * (H * W) + 
        tl.arange(0, H)[None, :, None] * W +
        tl.arange(0, W)[None, None, :],
        mask=mask[:, None, None] &
             (tl.arange(0, H)[None, :, None] < H) &
             (tl.arange(0, W)[None, None, :] < W),
        other=0.0
    )
    
    final_output = output_sigmoid_4d * multiply_data.to(output_sigmoid.dtype)
    
    # Store result
    tl.store(
        output_ptr + offsets[:, None, None] * (H * W) + 
        tl.arange(0, H)[None, :, None] * W +
        tl.arange(0, W)[None, None, :],
        final_output,
        mask=mask[:, None, None] &
             (tl.arange(0, H)[None, :, None] < H) &
             (tl.arange(0, W)[None, None, :] < W)
    )

@torch.fx.wrap
def linear_sigmoid_view_mul_fusion_b1(in_0, in_1, in_2, in_3):
    """
    Wrapper for batch size 1
    """
    assert in_2.shape[0] == 1, f"Expected batch size 1, got {in_2.shape[0]}"
    H, W = in_3.shape[2], in_3.shape[3]
    
    output = torch.empty((1, 64, H, W), dtype=in_3.dtype, device=in_3.device)
    
    BLOCK_SIZE = 64
    
    grid_n = (64 + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    linear_sigmoid_view_mul_kernel_b1[grid_n](
        in_0, in_1, in_2, in_3, output, H, W, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return linear_sigmoid_view_mul_fusion_b1