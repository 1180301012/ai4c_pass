import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def relu_flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU in-place (max(x, 0))
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu_flatten(x):
    # Handle different tensor shapes
    if x.dim() == 4 and x.size(2) == 1 and x.size(3) == 1:
        # Shape is [batch, channels, 1, 1] - optimize for this common case
        batch_size, channels = x.size(0), x.size(1)
        x_flat = x.reshape(-1)  # Flatten to [batch_size * channels]
        
        # Perform optimized ReLU on flattened tensor
        N = x_flat.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out_flat = torch.empty_like(x_flat)
        
        relu_flatten_kernel[(num_programs,)](
            x_ptr=x_flat,
            out_ptr=out_flat,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape back to [batch_size, channels]
        return out_flat.reshape(batch_size, channels)
    else:
        # Fallback for other shapes - use standard operations
        return torch.nn.functional.relu(x).flatten(1, -1)

def replacement_func():
    return optimized_relu_flatten