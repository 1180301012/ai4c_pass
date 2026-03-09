import torch
import triton
import triton.language as tl

# Pattern matching function: ReLU (inplace) followed by Sigmoid
def pattern(a, b):
    t = torch.nn.functional.relu(a, inplace=True) + b * 0.0  # Use b to avoid dead code
    out = torch.sigmoid(t)
    return out,

# Argument extraction function
def replacement_args(a, b):
    return (a, b)

# Triton kernel that fuses ReLU and Sigmoid operations
@triton.jit
def fused_relu_sigmoid_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse ReLU and Sigmoid operations
    # ReLU: x = max(0, x)
    relu_out = tl.maximum(x, 0.0)
    # Sigmoid: 1 / (1 + exp(-x))
    sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_activation_kernel(in_0):
    # Handle tensor shape - flatten spatial dimensions since they are 1x1
    if in_0.dim() == 4:
        # For shape [batch, channels, 1, 1], flatten to [batch * channels]
        tensor_2d = in_0.reshape(-1)
    else:
        tensor_2d = in_0
    
    N = tensor_2d.numel()
    
    # Choose block size based on tensor size for optimal performance
    if N < 1024:
        BLOCK_SIZE = 256
    elif N < 8192:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
        
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties as input
    out = torch.empty_like(in_0)
    out_2d = out.reshape(-1) if out.dim() == 4 else out
    
    # Launch the fused kernel
    fused_relu_sigmoid_kernel[(num_programs,)](
        in_ptr=tensor_2d,
        out_ptr=out_2d,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns the optimized kernel)
def replacement_func():
    return fused_activation_kernel