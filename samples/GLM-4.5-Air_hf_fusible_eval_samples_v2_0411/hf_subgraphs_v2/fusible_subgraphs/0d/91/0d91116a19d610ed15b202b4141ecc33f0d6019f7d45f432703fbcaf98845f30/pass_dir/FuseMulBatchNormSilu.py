import torch
import triton
import triton.language as tl

# Define the pattern that matches all variants of the computation
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern matches: (in_5 * in_4) → batch_norm → silu
    This pattern handles both verbose and concise coding styles from the models
    """
    # This captures the core computation structure used across all variants
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract all arguments needed for the fused kernel"""
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def fused_mul_batch_norm_silu_kernel(
    # Input pointers
    x_ptr,          # Main input tensor (in_5)
    sigmoid_ptr,    # Sigmoid output tensor (in_4)
    running_mean_ptr,    # Batch norm running mean (in_0)
    running_var_ptr,     # Batch norm running var (in_1)
    weight_ptr,     # Batch norm weight (in_3)
    bias_ptr,       # Batch norm bias (in_2)
    output_ptr,     # Final output
    
    # Tensor shapes and strides
    N, C, H, W,     # Input tensor dimensions
    sigmoid_stride_N, sigmoid_stride_C, sigmoid_stride_H, sigmoid_stride_W,
    x_stride_N, x_stride_C, x_stride_H, x_stride_W,
    output_stride_N, output_stride_C, output_stride_H, output_stride_W,
    
    # Batch norm parameters
    eps: tl.constexpr,
    momentum: tl.constexpr,
    
    # Block sizes
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """Fused kernel for: multiplication + batch normalization + SiLU activation"""
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate bounds for this thread
    n_start = pid_n * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    
    n_end = min(n_start + BLOCK_SIZE_N, N)
    c_end = min(c_start + BLOCK_SIZE_C, C)
    
    # Process spatial dimensions (H, W) for assigned batch and channel
    for h in range(H):
        for w in range(W):
            # Process a block of batch and channel dimensions
            for n in range(n_start, n_end):
                for c in range(c_start, c_end):
                    # Calculate memory addresses
                    sigmoid_offset = (n * sigmoid_stride_N + c * sigmoid_stride_C + 
                                     h * sigmoid_stride_H + w * sigmoid_stride_W)
                    x_offset = (n * x_stride_N + c * x_stride_C + 
                               h * x_stride_H + w * x_stride_W)
                    mean_offset = c  # running_mean is [C]
                    var_offset = c    # running_var is [C]
                    weight_offset = c  # weight is [C]
                    bias_offset = c    # bias is [C]
                    output_offset = (n * output_stride_N + c * output_stride_C + 
                                   h * output_stride_H + w * output_stride_W)
                    
                    # Load inputs
                    sigmoid_val = tl.load(sigmoid_ptr + sigmoid_offset)
                    x_val = tl.load(x_ptr + x_offset)
                    mean_val = tl.load(running_mean_ptr + mean_offset)
                    var_val = tl.load(running_var_ptr + var_offset)
                    weight_val = tl.load(weight_ptr + weight_offset)
                    bias_val = tl.load(bias_ptr + bias_offset)
                    
                    # Step 1: Element-wise multiplication
                    mul_val = x_val * sigmoid_val
                    
                    # Step 2: Batch normalization
                    # Normalize: (x - mean) / sqrt(var + eps)
                    normalized = (mul_val - mean_val) * rsqrt(var_val + eps)
                    
                    # Scale and shift: weight * normalized + bias
                    batch_norm_val = weight_val * normalized + bias_val
                    
                    # Step 3: SiLU activation: x * sigmoid(x)
                    silu_val = batch_norm_val * tl.sigmoid(batch_norm_val)
                    
                    # Store result
                    tl.store(output_ptr + output_offset, silu_val)

@torch.fx.wrap
def fused_mul_batch_norm_silu(in_0, in_1, in_2, in_3, in_4, in_5):
    """Wrapper function to launch the fused kernel"""
    # Get tensor shapes and properties
    x_shape = in_5.shape  # [N, C, H, W]
    sigmoid_shape = in_4.shape  # [N, C, 1, 1] or similar
    
    N, C, H, W = x_shape
    
    # Get strides for all tensors
    x_stride = in_5.stride()
    sigmoid_stride = in_4.stride()
    
    # For batch norm parameters, they are 1D [C], so stride is just [1]
    mean_stride = (1,)
    var_stride = (1,)
    weight_stride = (1,)
    bias_stride = (1,)
    
    # For output tensor, create it with same properties as input
    output = torch.empty_like(in_5)
    output_stride = output.stride()
    
    # Determine optimal block sizes based on tensor dimensions
    # We want good GPU occupancy, so balance block sizes
    BLOCK_SIZE_N = max(1, min(64, N // 4))  # Process 4 worth of batches
    BLOCK_SIZE_C = max(1, min(256, C // 4))  # Process 4 worth of channels
    
    # Calculate grid size
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch Triton kernel
    fused_mul_batch_norm_silu_kernel[(
        grid_n,
        grid_c,
        1  # No third dimension for spatial blocks - we handle those in kernel
    )](
        # Input pointers
        in_5, in_4, in_0, in_1, in_3, in_2, output,
        
        # Tensor shapes
        N, C, H, W,
        
        # Strides
        sigmoid_stride[0], sigmoid_stride[1], sigmoid_stride[2], sigmoid_stride[3],
        x_stride[0], x_stride[1], x_stride[2], x_stride[3],
        output_stride[0], output_stride[1], output_stride[2], output_stride[3],
        
        # Batch norm parameters
        1e-05, 0.1,  # eps, momentum
        
        # Block sizes
        BLOCK_SIZE_N, BLOCK_SIZE_C,
    )
    
    return output

def replacement_func():
    """Return the fused function - no arguments allowed"""
    return fused_mul_batch_norm_silu