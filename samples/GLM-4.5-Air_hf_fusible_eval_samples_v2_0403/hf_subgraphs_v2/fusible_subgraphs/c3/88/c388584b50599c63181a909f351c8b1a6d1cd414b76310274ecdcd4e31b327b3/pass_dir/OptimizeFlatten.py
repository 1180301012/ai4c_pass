import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern: Optimize flatten operation for [N, C, 1, 1] tensors"""
    # Flatten operation from dimension 1 to -1
    flattened = x.flatten(1, -1)
    return flattened

@triton.jit
def optimized_flatten_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized flatten kernel for [N, C, 1, 1] tensors"""
    # Each program processes one output element
    pid = tl.program_id(0)
    total_elements = N * C * 1 * 1
    
    if pid < total_elements:
        # Calculate indices in original tensor
        n = pid // (C * 1 * 1)
        c = (pid % (C * 1 * 1)) // (1 * 1)
        h = (pid % (1 * 1)) // 1
        w = pid % 1
        
        # Load from original location
        src_idx = n * C * H * W + c * H * W + h * W + w
        x_val = tl.load(x_ptr + src_idx)
        
        # Store in flattened location (since [N, C, 1, 1] -> [N*C], just copy)
        tl.store(out_ptr + pid, x_val)

@torch.fx.wrap
def optimized_flatten(x):
    """Optimized flatten for tensors with H=W=1"""
    N, C, H, W = x.shape
    
    # For [N, C, 1, 1] tensors, flatten is just a reshape
    # But we can make it explicit to avoid any overhead
    if H == 1 and W == 1:
        # In this case, flatten is essentially just a view operation
        # but we can optimize by using a simple kernel
        flattened_shape = (N * C,)
        out = torch.empty(flattened_shape, dtype=x.dtype, device=x.device)
        
        # For this specific case where H=W=1, we can optimize the kernel
        BLOCK_SIZE = 1024
        num_programs = (N * C + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        optimized_flatten_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            N=N, C=C, H=H, W=W,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
    else:
        # For general case, fall back to standard flatten
        return x.flatten(1, -1)

def replacement_args(x):
    """Extract arguments for flatten operation"""
    return (x,)

def replacement_func():
    """Return the optimized flatten function"""
    return optimized_flatten