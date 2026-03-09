import torch
import triton
import triton.language as tl

@triton.jit
@triton.autotune(
    configs=[
        triton.Config(num_warps=4, num_stages=2),
        triton.Config(num_warps=8, num_stages=2),
        triton.Config(num_warps=4, num_stages=3),
        triton.Config(num_warps=8, num_stages=3),
    ],
    key=['n', 'c_in', 'c_out', 'h_out', 'w_out'],
)
@triton.heuristics({
    "use_vector_load": lambda args: args['w_out'] >= 16,
})
def simple_conv_kernel(input_ptr, weight_ptr, bias_ptr, out_ptr, n, c_in, c_out, h_out, w_out, use_vectorize: tl.constexpr):
    # Vectorization configuration
    BLOCK_SIZE_M = 16  # Block size for spatial dimensions
    
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate output spatial position
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    h_offsets = (m_offsets // w_out) % h_out
    w_offsets = m_offsets % w_out
    mask = m_offsets < h_out * w_out
    
    # Process each output channel
    for c_out_idx in range(pid_c * 1, min((pid_c + 1) * 1, c_out)):
        # Load bias
        bias_val = tl.load(bias_ptr + c_out_idx)
        
        # Initialize output tile
        if use_vectorize:
            # Vectorized loads for spatial dimensions
            output_vals = tl.full((BLOCK_SIZE_M,), bias_val, dtype=tl.float32)
            
            # Outer loop over input channels
            for c_in_idx in range(0, c_in, 4):  # Process 4 channels at a time
                # Load weight vector
                weight_offset = c_out_idx * c_in + c_in_idx
                weights = tl.load(weight_ptr + weight_offset).to(tl.float32)
                
                # Load input vector
                input_offset = (pid_n * c_in * h_out * w_out + 
                              c_in_idx * h_out * w_out + 
                              pid_m * BLOCK_SIZE_M)
                inputs = tl.load(input_ptr + input_offset).to(tl.float32)
                
                # Vector multiply-add
                output_vals += inputs * weights
        else:
            # Scalar operations for small tensors
            for spatial_idx in range(BLOCK_SIZE_M):
                if pid_m * BLOCK_SIZE_M + spatial_idx < h_out * w_out:
                    h = (pid_m * BLOCK_SIZE_M + spatial_idx) // w_out
                    w = (pid_m * BLOCK_SIZE_M + spatial_idx) % w_out
                    
                    bias_val = tl.load(bias_ptr + c_out_idx)
                    output_val = bias_val
                    
                    for c_in_idx in range(c_in):
                        weight_offset = c_out_idx * c_in + c_in_idx
                        weight = tl.load(weight_ptr + weight_offset)
                        
                        input_offset = (pid_n * c_in * h_out * w_out + 
                                      c_in_idx * h_out * w_out + 
                                      h * w_out + w)
                        input_val = tl.load(input_ptr + input_offset)
                        
                        output_val += input_val * weight
                    
                    # Store result
                    out_offset = (pid_n * c_out * h_out * w_out + 
                                 c_out_idx * h_out * w_out + 
                                 h * w_out + w)
                    tl.store(out_ptr + out_offset, output_val)

@torch.fx.wrap
def simple_conv_optimized(input_tensor, weight_tensor, bias_tensor):
    # Get dimensions
    N = input_tensor.shape[0]           # Batch size: 1 or 32
    C_in = input_tensor.shape[1]        # Input channels: 64
    H_out = input_tensor.shape[2]       # Height: 20
    W_out = input_tensor.shape[3]       # Width: 20
    C_out = weight_tensor.shape[0]      # Output channels: 1
    
    # Create output with correct shape
    output = torch.empty((N, C_out, H_out, W_out), dtype=torch.float32, device="cuda")
    
    # Set 3D grid dimensions for better GPU occupancy
    grid_h = (H_out * W_out + 15) // 16  # Spatial blocks
    grid_n = N                           # Batch dimension
    grid_c = (C_out + 0) // 1           # Output channel blocks (1 block per channel)
    grid = (grid_h, grid_n, grid_c)
    
    # Launch kernel
    simple_conv_kernel[grid](
        input_tensor,
        weight_tensor.flatten(),  # weight is [1, 64, 1, 1] -> flatten to [64]
        bias_tensor,              # bias is [1]
        output,
        N, C_in, C_out, H_out, W_out
    )
    
    # Apply view operation: [N, 1, 20, 20] -> [N, 1, 400]
    return output.view(N, 1, -1)

def pattern(x, y, z):
    w = torch.conv2d(x, y, z, (1, 1), (0, 0), (1, 1), 1)
    return w

def replacement_args(x, y, z):
    return (x, y, z)

def replacement_func():
    return simple_conv_optimized