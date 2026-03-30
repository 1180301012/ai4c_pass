import torch
import triton
import triton.language as tl

def pattern(input_3, input_1, input_0):
    """
    Pattern matching: torch.nn.functional.linear + permute(0, 3, 1, 2)
    
    This fuses a linear transformation followed by dimension permutation.
    Original: linear -> permute
    Optimized: Single kernel that directly computes permuted result
    """
    linear = torch.nn.functional.linear(input_3, input_1, input_0)
    result = linear.permute(0, 3, 1, 2)
    return result

def replacement_args(input_3, input_1, input_0):
    """
    Extract arguments for the optimized kernel
    """
    return (input_3, input_1, input_0)

@triton.jit
def linear_permute_kernel(
    input_ptr,      # [1, 196, 196, 3]
    weight_ptr,     # [16, 3] 
    bias_ptr,       # [16]
    output_ptr,     # [1, 16, 196, 196]
    M,              # 1 (batch size)
    N,              # 16 (output features)
    
    # Constants for dimensions
    H_in: tl.constexpr,   # 196 (input height)
    W_in: tl.constexpr,   # 196 (input width) 
    C_in: tl.constexpr,   # 3 (input channels)
    HW: tl.constexpr,     # H_in * W_in (total spatial positions)
    
    BLOCK_SIZE_HW: tl.constexpr
):
    """
    Optimized kernel that fuses linear transformation with dimension permutation.
    Directly computes output in [1, 16, 196, 196] layout instead of [1, 196, 196, 16].
    """
    
    # Compute program IDs: [batch, output_channel, spatial_position]
    b_id = tl.program_id(0)      # batch index (0 for batch size 1)
    n_id = tl.program_id(1)      # output channel (0-15)  
    hw_id = tl.program_id(2)     # spatial position in flattened HW
    
    # Check bounds - avoid chained boolean operators
    if b_id >= M:
        return
    if n_id >= N:
        return
    if hw_id >= HW:
        return
    
    # Convert flattened hw_id back to h, w coordinates
    h_id = hw_id // W_in
    w_id = hw_id % W_in
    
    # Input position: [b_id, h_id, w_id, :]
    input_base = input_ptr + (b_id * HW * C_in + hw_id * C_in)
    
    # Load input slice at [h_id, w_id] with all input channels
    # Workaround: use power-of-2 arange and mask
    input_slice = tl.load(input_base + tl.arange(0, 4), mask=tl.arange(0, 4) < C_in)
    
    # Load weights for this output channel: [N, C_in] -> load row n_id
    weights = tl.load(weight_ptr + n_id * C_in + tl.arange(0, 4), mask=tl.arange(0, 4) < C_in)
    
    # Load bias for this output channel
    bias = tl.load(bias_ptr + n_id)
    
    # Compute dot product: output = input_slice @ weights + bias
    result = tl.sum(input_slice * weights) + bias
    result = result.to(input_ptr.dtype.element_ty)
    
    # Store output in permuted layout [1, 16, 196, 196]
    # Output index: [b_id, n_id, h_id, w_id]
    output_idx = b_id * N * HW + n_id * HW + hw_id
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def fused_linear_permute(input_3, weight_1, bias_0):
    """
    Wrapper function that launches the fused linear+permute kernel
    """
    M, H_in, W_in, C_in = input_3.shape
    N = weight_1.shape[0]  # output features = 16
    
    # Output should be [1, 16, 196, 196]
    output_shape = [M, N, H_in, W_in]
    output = torch.empty(output_shape, dtype=input_3.dtype, device=input_3.device)
    
    # Flattened spatial dimensions
    HW = H_in * W_in
    
    # Launch kernel with 3D grid: [batch, output_channels, spatial_positions]
    grid = (M, N, HW)
    
    linear_permute_kernel[grid](
        input_3,
        weight_1,
        bias_0,
        output,
        M,
        N,
        H_in, W_in, C_in, HW,
        BLOCK_SIZE_HW=1  # Process each spatial position individually
    )
    
    return output

def replacement_func():
    """
    Return the optimized function reference
    """
    return fused_linear_permute