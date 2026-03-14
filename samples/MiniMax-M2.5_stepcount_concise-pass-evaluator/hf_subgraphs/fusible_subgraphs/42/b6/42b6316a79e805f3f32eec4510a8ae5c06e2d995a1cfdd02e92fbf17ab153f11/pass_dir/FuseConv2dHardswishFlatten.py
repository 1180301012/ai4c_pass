import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match Conv2d + Hardswish + Flatten pattern
    - Conv2d with 1x1 kernel, stride=1, padding=0, dilation=1, groups=1
    - Hardswish activation (inplace=True)
    - Flatten from dimension 1
    
    The pattern must exactly mirror model.py operations.
    """
    # Conv2d: in_2 (input) * in_1 (weight) + in_0 (bias)
    # Arguments: input, weight, bias, stride, padding, dilation, groups
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # Hardswish activation with inplace=True
    tmp_3 = torch.nn.functional.hardswish(tmp_2, True)
    # Flatten from dimension 1
    tmp_4 = tmp_3.flatten(1, -1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2)


# Autotune configurations optimized for the specific problem sizes
# Graph 1: N=128, K=1280, C=960 -> 163840 elements
# Graph 2: N=1, K=1280, C=960 -> 1280 elements
@triton.autotune(
    configs=[
        # Configs optimized for large K (1280) with varying N
        # Using larger BLOCK_SIZE_K for better memory coalescing
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 256}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 512}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 128}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 128}, num_stages=1, num_warps=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def fused_conv1x1_hardswish_kernel(
    # Pointers
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Problem dimensions
    N: tl.constexpr,  # batch size
    K: tl.constexpr,  # output channels
    C: tl.constexpr,  # input channels
    # Block sizes (determined by autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    """
    Optimized fully fused 1x1 Conv + Hardswish kernel.
    
    Uses blocked matrix multiplication with better memory access patterns.
    """
    # Compute the starting row and column for this program
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    
    # Combine program IDs for 2D grid
    pid_m = pid % num_pid_m
    pid_k = pid // num_pid_m
    
    # Compute offsets
    row_offset = pid_m * BLOCK_M
    col_offset = pid_k * BLOCK_K
    
    # Create ranges and masks
    row_indices = row_offset + tl.arange(0, BLOCK_M)
    col_indices = col_offset + tl.arange(0, BLOCK_K)
    row_mask = row_indices < N
    col_mask = col_indices < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    # Loop over reduction dimension (input channels)
    # This implements the matrix multiplication: input @ weight.T
    for c in range(0, C, 1):
        # Load input tile: shape [BLOCK_M]
        input_indices = row_indices * C + c
        input_vals = tl.load(input_ptr + input_indices, mask=row_mask, other=0.0)
        
        # Load weight tile: shape [BLOCK_K]
        # Weight is stored as [K, C], so we compute col_idx * C + c
        weight_indices = col_indices * C + c
        weight_vals = tl.load(weight_ptr + weight_indices, mask=col_mask, other=0.0)
        
        # Accumulate outer product: [BLOCK_M, 1] * [1, BLOCK_K] -> [BLOCK_M, BLOCK_K]
        accumulator += input_vals[:, None] * weight_vals[None, :]
    
    # Add bias
    bias_vals = tl.load(bias_ptr + col_indices, mask=col_mask, other=0.0)
    accumulator = accumulator + bias_vals
    
    # Apply hardswish: hswish(x) = x * ReLU6(x + 3) / 6
    x_plus_3 = accumulator + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    result = accumulator * relu6 / 6.0
    
    # Store output
    output_indices = row_indices[:, None] * K + col_indices[None, :]
    output_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(output_ptr + output_indices, result, mask=output_mask)


@torch.fx.wrap
def fused_conv_hardswish_flatten(bias, weight, input):
    """
    Fully fused kernel: 1x1 Conv2d + Hardswish + Flatten
    
    Input shape: [N, C, 1, 1]
    Weight shape: [K, C, 1, 1]
    Bias shape: [K]
    Output shape: [N, K]
    """
    N = input.shape[0]
    C = input.shape[1]
    K = weight.shape[0]
    
    # Flatten input [N, C, 1, 1] -> [N, C]
    input_flat = input.view(N, C)
    
    # Flatten weight [K, C, 1, 1] -> [K, C]
    weight_flat = weight.squeeze(-1).squeeze(-1)
    
    # Allocate output
    output = torch.empty((N, K), dtype=torch.float32, device=input.device)
    
    # Grid: enough programs to cover all output elements
    grid = lambda M: (triton.cdiv(N, M['BLOCK_M']) * triton.cdiv(K, M['BLOCK_K']),)
    
    fused_conv1x1_hardswish_kernel[grid](
        input_ptr=input_flat,
        weight_ptr=weight_flat,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        K=K,
        C=C
    )
    
    return output


def replacement_func():
    """Return the replacement function"""
    return fused_conv_hardswish_flatten