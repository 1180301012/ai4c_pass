import torch
import triton
import triton.language as tl

# Pattern matching function - matches the fused computation chain
def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel - fused sigmoid + view + mul + add + ReLU  
@triton.jit
def fused_sigmoid_mul_add_relu_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate starting position in the flattened tensor [N, C, H, W]
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < (N * C * H * W)
    
    # For each index, compute channel and position
    # idx = n * C * H * W + c * H * W + h * W + w
    # We know N=1, so: idx = c * H * W + h * W + w
    linear_idx = indices % (N * C * H * W)
    c = (linear_idx // (H * W)) % C  # channel index
    spatial_idx = linear_idx % (H * W)  # spatial index within channel
    
    # Load corresponding sigmoid weight for this channel
    sigmoid_weight = tl.load(in_0_ptr + c, mask=mask)
    
    # Load input tensor value at this position
    in_1_val = tl.load(in_1_ptr + linear_idx, mask=mask)
    
    # Compute fused operation: in_1 * (1 + sigmoid_weight)
    # This is mathematically equivalent to: in_1 + in_1 * sigmoid_weight
    fused_val = in_1_val * (1.0 + sigmoid_weight)
    
    # Apply ReLU activation
    relu_val = tl.maximum(fused_val, 0.0)
    
    # Store result
    tl.store(out_ptr + indices, relu_val, mask=mask)

# Kernel wrapper function
@torch.fx.wrap
def fused_sigmoid_mul_add_relu(in_0, in_1):
    # Get tensor shapes
    N, C, H, W = in_1.shape
    total_elements = N * C * H * W
    
    # Set up Triton kernel parameters
    BLOCK_SIZE = 1024  # Optimized block size for GPU occupancy
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Launch Triton kernel with 1D grid
    fused_sigmoid_mul_add_relu_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns the optimized function)
def replacement_func():
    return fused_sigmoid_mul_add_relu