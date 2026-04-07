import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """Pattern: ReLU → Sigmoid sequence."""
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1

# Argument extraction function  
def replacement_args(in_0):
    return (in_0,)

# Optimized fused kernel
@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Sigmoid kernel."""
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: ReLU -> Sigmoid
    # ReLU: max(0, x)
    relu_out = tl.maximum(x, 0.0)
    # Sigmoid: 1 / (1 + exp(-relu_out))
    sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store output
    tl.store(y_ptr + offsets, sigmoid_out, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid(x):
    """Wrapper function for fused kernel."""
    N = x.numel()
    BLOCK_SIZE = 1024  # Can be autotuned
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=x.dtype)
    
    # Handle different data types
    if x.dtype == torch.float16:
        # For float16, use a more careful implementation
        @triton.jit
        def fused_relu_sigmoid_fp16(
            x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # ReLU
            relu_out = tl.maximum(x, 0.0)
            # Sigmoid with better numerical stability
            sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
            
            tl.store(y_ptr + offsets, sigmoid_out, mask=mask)
            
        fused_relu_sigmoid_fp16[(num_programs,)](
            x_ptr=x,
            y_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif x.dtype == torch.bfloat16:
        # For bfloat16, use similar approach as float32
        @triton.jit
        def fused_relu_sigmoid_bf16(
            x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # ReLU
            relu_out = tl.maximum(x, 0.0)
            # Sigmoid
            sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
            
            tl.store(y_ptr + offsets, sigmoid_out, mask=mask)
            
        fused_relu_sigmoid_bf16[(num_programs,)](
            x_ptr=x,
            y_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:  # float32
        fused_relu_sigmoid_kernel[(num_programs,)](
            x_ptr=x,
            y_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function (returns function reference, no arguments)
def replacement_func():
    return fused_relu_sigmoid