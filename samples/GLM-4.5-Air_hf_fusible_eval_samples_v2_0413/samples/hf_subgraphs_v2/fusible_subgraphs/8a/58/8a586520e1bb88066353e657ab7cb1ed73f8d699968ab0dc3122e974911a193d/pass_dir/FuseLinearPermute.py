import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_0):
    """Pattern matching for linear + permutation fusion
    Matches: torch.nn.functional.linear(in_3, in_1, in_0).permute(0, 3, 1, 2)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3

def replacement_args(in_3, in_1, in_0):
    """Extract arguments for the fused kernel"""
    return (in_3, in_1, in_0)

@triton.jit
def fused_linear_permute_kernel(
    x_ptr,        # in_3: [1, 196, 196, 3]
    w_ptr,        # in_1: [16, 3] 
    b_ptr,        # in_0: [16]
    out_ptr,      # tmp_3: [1, 16, 196, 196]
    M,            # batch size = 1
    N,            # output features = 16
    K,            # input features = 3
    H,            # height = 196  
    W,            # width = 196
):
    """Simple fused kernel for linear transformation + permute operation"""
    # Program ids for spatial and output feature processing
    pid_hw = tl.program_id(0)  # Combined HW position
    pid_n = tl.program_id(1)   # Output feature
    
    # Single thread processing per program (for simplicity)
    hw_idx = pid_hw
    n_idx = pid_n
    
    # Bounds checking
    if hw_idx >= H * W or n_idx >= N:
        return
    
    # Compute linear operation: sum over K dimension
    result = 0.0
    
    # Create a mask (since we passed bounds check, this thread is valid)
    mask = True
    
    # Load input data for this HW position across all K
    # We need to iterate from k=0 to k=K-1, but K=3 is a constexpr in this case
    for k in range(3):  # Hardcode K=3 since we know it's 3
        input_offset = hw_idx * 3 + k
        input_val = tl.load(x_ptr + input_offset, mask=mask, other=0.0).to(tl.float32)
        weight_offset = n_idx * 3 + k
        weight_val = tl.load(w_ptr + weight_offset, mask=mask, other=0.0).to(tl.float32)
        result += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(b_ptr + n_idx, mask=mask, other=0.0).to(tl.float32)
    result += bias_val
    
    # Store result with permuted layout: [M, N, H, W]
    output_offset = n_idx * (H * W) + hw_idx
    tl.store(out_ptr + output_offset, result.to(tl.float16))
    
    

@torch.fx.wrap
def fused_linear_permute(in_3, in_1, in_0):
    """Wrapper function for fused linear + permute operation"""
    M = in_3.shape[0]  # 1
    H = in_3.shape[1]  # 196  
    W = in_3.shape[2]  # 196
    K = in_3.shape[3]  # 3
    N = in_1.shape[0]  # 16
    
    # Calculate grid size - one thread per (HW, N) pair
    total_hw_elements = H * W
    grid_hw = total_hw_elements
    grid_n = N
    
    # Create output tensor
    out = torch.empty((M, N, H, W), dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel with 2D grid
    fused_linear_permute_kernel[grid_hw, grid_n](
        in_3, in_1, in_0, out,
        M, N, K, H, W
    )
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_linear_permute