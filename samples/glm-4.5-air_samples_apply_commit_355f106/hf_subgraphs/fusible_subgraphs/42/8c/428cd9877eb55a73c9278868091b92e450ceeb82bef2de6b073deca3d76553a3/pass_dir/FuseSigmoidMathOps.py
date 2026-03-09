import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_math_ops_kernel(
    input_ptr,
    out_ptr,
    N, C, L,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Calculate output coordinates
    out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    out_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    # Combine indices
    indices = (out_n[:, None, None] * (C * L) + 
               out_c[None, :, None] * L + 
               out_l[None, None, :]).to(tl.int64)
    
    # Flatten indices for loading
    linear_indices = indices.flatten()
    
    # Load input values
    x = tl.load(input_ptr + linear_indices, 
                mask=(linear_indices < (N * C * L)).to(tl.int64))
    
    # Compute fused operations: sigmoid(x) - 0.25 * 3.141592653589793
    # Step 1: Sigmoid
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    
    # Step 2: Subtract 0.25
    result = sigmoid_x - 0.25
    
    # Step 3: Multiply by pi
    pi = 3.141592653589793
    final_result = result * pi
    
    # Store results
    tl.store(out_ptr + linear_indices, final_result, 
             mask=(linear_indices < (N * C * L)).to(tl.int64))

@torch.fx.wrap
def sigmoid_math_ops_optimized(input_tensor):
    N, C, L = input_tensor.shape
    
    # Set grid dimensions
    grid = (
        (N + 7) // 8,    # N dimension blocks
        (C + 15) // 16,  # C dimension blocks
        (L + 15) // 16,  # L dimension blocks
    )
    
    # Create output tensor
    output = torch.empty((N, C, L), dtype=torch.float32, device="cuda")
    
    # Launch kernel
    sigmoid_math_ops_kernel[grid](
        input_tensor,
        output,
        N, C, L,
    )
    
    return output

def sigmoid_math_pattern(input_tensor):
    # Sigmoid operation
    sigmoid_output = input_tensor.sigmoid()
    
    # Arithmetic operations
    subtracted = sigmoid_output - 0.25
    multiplied = subtracted * 3.141592653589793
    
    return sigmoid_output, subtracted, multiplied

def replacement_args(input_tensor):
    return (input_tensor,)

def replacement_func():
    return sigmoid_math_ops_optimized