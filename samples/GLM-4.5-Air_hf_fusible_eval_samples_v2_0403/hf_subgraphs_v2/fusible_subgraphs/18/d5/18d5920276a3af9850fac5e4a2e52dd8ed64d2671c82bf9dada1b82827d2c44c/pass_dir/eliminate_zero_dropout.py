import torch
import triton
import triton.language as tl

def pattern(x):
    # Match just the flatten operation
    return (x.flatten(1, -1),)

def replacement_args(x):
    return (x,)

@triton.jit
def relu_flatten_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the flattened output
    idx = tl.program_id(0)
    mask = idx < N * C
    
    # Calculate coordinates
    batch_idx = idx // C
    channel_idx = idx % C
    
    # Load input element properly using strides
    # For tensor [N, C, H, W] where H=W=1, stride is [C*H*W, H*W, W, 1] = [C, 1, 1, 1]
    x_offset = batch_idx * C + channel_idx
    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    relu_val = tl.maximum(x_val, 0.0)
    
    # Store result to [N, C] flattened output
    tl.store(out_ptr + idx, relu_val, mask=mask)

@torch.fx.wrap
def optimized_relu_flatten(x):
    N, C, H, W = x.shape
    
    # Create output tensor with flattened shape [N, C]
    out = torch.empty((N, C), dtype=x.dtype, device=x.device)
    
    # Calculate grid size - use larger block size for better occupancy
    total_elements = N * C
    BLOCK_SIZE = 2048  # Increased from 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_relu_flatten