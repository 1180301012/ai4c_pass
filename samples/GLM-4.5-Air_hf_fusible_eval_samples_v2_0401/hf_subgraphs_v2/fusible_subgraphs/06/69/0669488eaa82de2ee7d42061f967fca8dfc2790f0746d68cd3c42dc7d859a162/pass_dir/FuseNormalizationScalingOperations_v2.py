import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    # The complete computation pattern with the alternative scaling constant
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322  # Alternative scaling constant
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7

def replacement_args(in_0, in_1):
    # Extract the necessary arguments for the optimized kernel
    # Calculate flattened shape manually to avoid forbidden torch operations
    original_shape = in_1.shape
    # flatten(dim=2) means we concatenate from dim 2 onwards
    N = original_shape[0] * original_shape[1]
    M = original_shape[2] * original_shape[3]
    
    return (in_0, in_1, N, M)

# Optimized Triton kernel that fuses ReLU + Flatten + Norm + Scale + Clamp + Division + Multiplication
@triton.jit
def fused_normalization_kernel(
    # Input tensors
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    
    # Tensor metadata
    N,  # First dimension after flattening (num_batches * num_features)
    M,  # Second dimension after flattening (flattened spatial dimensions)
    
    # Scaling parameters
    scale_const,
    epsilon,
    
    # Block sizes from autotune
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Iterate over M dimension
    pid_n = tl.program_id(1)  # Iterate over N dimension
    
    # Compute memory addresses
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Load input data
    in_1_ptrs = in_1_ptr + (n_offsets[:, None] * M + m_offsets[None, :]).to(tl.int64)
    in_1 = tl.load(in_1_ptrs, mask=m_mask[None, :] & n_mask[:, None], other=0.0)
    
    # Load scale factor (in_0). This should be broadcastable
    if in_0_ptr is not None:
        # Assuming in_0 is a scalar or broadcastable tensor
        scale_ptrs = in_0_ptr + (n_offsets[:, None] * M + m_offsets[None, :]).to(tl.int64)
        scale_factor = tl.load(scale_ptrs, mask=m_mask[None, :] & n_mask[:, None], other=1.0)
    else:
        scale_factor = 1.0
    
    # Step 1: Apply ReLU activation (fused)
    relu_out = tl.maximum(in_1, 0.0)
    
    # Step 2: Compute L2 norm along M dimension (equivalent to original dim=-1)
    # We need to compute sum of squares first, then take square root, per column
    square = relu_out * relu_out
    
    # Sum squares along M dimension for each column
    sum_squares = tl.sum(square, axis=1)
    
    # Add epsilon for numerical stability (fused clamping)
    sum_squares = tl.maximum(sum_squares, epsilon * epsilon)
    norm = tl.sqrt(sum_squares)
    
    # Reshape norm to broadcast with relu_out: [N, 1]
    norm = norm.reshape([N, 1])
    
    # Step 3: Apply scaling constant and clamping (fused)
    scaled_norm = norm * scale_const
    clamped_norm = tl.maximum(scaled_norm, epsilon)
    
    # Step 4: Normalize by the norm and apply scaling factor
    # Avoid division by zero
    inv_norm = 1.0 / clamped_norm
    normalized = relu_out * inv_norm
    
    # Apply final scaling by in_0
    output = normalized * scale_factor
    
    # Store result
    out_ptrs = out_ptr + (n_offsets[:, None] * M + m_offsets[None, :]).to(tl.int64)
    tl.store(out_ptrs, output, mask=m_mask[None, :] & n_mask[:, None])

# Simple wrapper without autotune to avoid config issues
@torch.fx.wrap
def fused_normalization_forward_simple(in_0, in_1, N, M, scale_const=0.07216878364870322, epsilon=1e-05):
    # Use conservative block sizes that work for most data types
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 256
    
    # Calculate grid size
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    dtype = in_1.dtype
    out = torch.empty(N, M, dtype=dtype, device=in_1.device)
    
    # Determine if in_0 exists and is valid
    if in_0 is None:
        in_0_ptr = None
    else:
        in_0_ptr = in_0.data_ptr()
    
    # Launch the kernel
    fused_normalization_kernel[(grid_m, grid_n)](
        in_0_ptr=in_0_ptr,
        in_1_ptr=in_1.data_ptr(),
        out_ptr=out.data_ptr(),
        N=N,
        M=M,
        scale_const=scale_const,
        epsilon=epsilon,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

@torch.fx.wrap
def fused_normalization_forward(in_0, in_1, scale_const=0.07216878364870322, epsilon=1e-05):
    # Get input shape manually (avoiding torch.flatten)
    original_shape = in_1.shape
    N = original_shape[0] * original_shape[1]
    M = original_shape[2] * original_shape[3]
    
    return fused_normalization_forward_simple(in_0, in_1, N, M, scale_const, epsilon)

def replacement_func():
    def optimized_forward(in_0, in_1, *args):
        # args should contain (N, M) from replacement_args
        N, M = args
        # Use the alternative scaling constant for this pass
        return fused_normalization_forward_simple(in_0, in_1, N, M, scale_const=0.07216878364870322)
    
    return optimized_forward