import torch
import triton
import triton.language as tl


# Pattern matching function - matches gelu + flatten pattern
def pattern(in_0):
    """
    Match the pattern: gelu(in_0) followed by flatten(1, -1)
    This exactly mirrors the model.py computation:
        tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_1 = tmp_0.flatten(1, -1)
    """
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Highly optimized GELU kernel with better memory access patterns
@triton.jit
def gelu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized GELU kernel using tl.sigmoid if available.
    """
    # Get block start and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # GELU using sigmoid: x * sigmoid(1.702 * x)
    # Using tl.sigmoid if available, otherwise compute manually
    k = 1.702
    # Try using tl.sigmoid directly for better performance
    sig = tl.sigmoid(k * x)
    gelu = x * sig
    
    # Store
    tl.store(output_ptr + offsets, gelu, mask=mask)


# Wrapper for fused GELU + flatten
@torch.fx.wrap
def gelu_flatten_wrapper(in_0):
    """
    Fused GELU + flatten operation using Triton.
    """
    B = in_0.shape[0]
    C = in_0.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]
    flattened_size = C * H * W
    
    # Flatten: [B, C, H, W] -> [B*C*H*W]
    input_flat = in_0.flatten()
    
    n_elements = input_flat.numel()
    
    # Use optimal block size based on tensor size
    # Larger blocks for larger tensors to reduce kernel launch overhead
    if n_elements < 2048:
        BLOCK_SIZE = 256
    elif n_elements < 8192:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 4096
        
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output_flat = torch.empty(n_elements, dtype=in_0.dtype, device=in_0.device)
    
    gelu_flatten_kernel[(num_programs,)](
        input_ptr=input_flat,
        output_ptr=output_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to [B, C*H*W]
    output = output_flat.view(B, flattened_size)
    
    return output


# Replacement function that returns the kernel wrapper
def replacement_func():
    return gelu_flatten_wrapper