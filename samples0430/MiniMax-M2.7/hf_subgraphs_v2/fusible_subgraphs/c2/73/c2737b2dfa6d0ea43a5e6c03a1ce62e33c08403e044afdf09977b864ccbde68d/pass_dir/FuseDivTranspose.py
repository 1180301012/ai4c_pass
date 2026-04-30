import torch
import triton
import triton.language as tl

def pattern(in_0, const_value):
    """
    Match pattern: in_0 / const_value followed by transpose(-1, -2)
    """
    tmp_0 = in_0 / const_value
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1

def replacement_args(in_0, const_value):
    # Extract the constant value (it's a float)
    return (in_0, const_value)

@torch.fx.wrap
def kernel_wrapper(in_0, const_value):
    """
    Wrapper function that launches the Triton kernel.
    The transpose(-1, -2) means we swap the last two dimensions.
    We perform division and write directly to the transposed output layout.
    """
    # Get input shape
    shape = in_0.shape
    ndim = len(shape)
    
    # For transpose(-1, -2), we swap the last two dimensions
    # Original shape: [..., H, W]
    # Transposed shape: [..., W, H]
    H = shape[-2]
    W = shape[-1]
    
    # Create output with transposed shape
    new_shape = list(shape[:-2]) + [W, H]
    output = torch.empty(new_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate total number of output elements = outer * W * H
    outer_elements = 1
    for dim_size in shape[:-2]:
        outer_elements *= dim_size
    total_output_elements = outer_elements * W * H
    
    # Compute reciprocal of scale for multiplication (faster than division)
    scale_recip = 1.0 / const_value
    
    # Precompute constants for the kernel
    hw_size = H * W
    
    # Launch the fused kernel
    grid = lambda meta: (total_output_elements,)
    
    # Use autotuned kernel
    fuse_div_transpose_kernel_autotuned[grid](
        input_ptr=in_0,
        output_ptr=output,
        scale_recip=scale_recip,
        H=H,
        W=W,
        outer_elements=outer_elements,
        hw_size=hw_size,
        total_output_elements=total_output_elements,
    )
    
    return output

# Auto-tuned kernel configuration
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 4096, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 8192, 'num_warps': 16}),
    ],
    key=['total_output_elements'],
)
@triton.jit
def fuse_div_transpose_kernel_autotuned(
    input_ptr,
    output_ptr,
    scale_recip: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    outer_elements: tl.constexpr,
    hw_size: tl.constexpr,
    total_output_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs division by constant and transpose(-1, -2) in one pass.
    
    Uses vectorized loads for better memory bandwidth utilization.
    """
    pid = tl.program_id(0)
    
    # Calculate range of output elements for this program
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, total_output_elements)
    
    # Use a mask for safe memory access
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_output_elements
    
    # Convert output linear index to (outer_idx, w, h) coordinates
    outer_idx = offs // (W * H)
    remainder = offs % (W * H)
    w = remainder // H
    h = remainder % H
    
    # Compute input linear indices - contiguous read from input
    input_offs = outer_idx * hw_size + h * W + w
    
    # Load, apply multiplication, and store
    vals = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    vals = vals * scale_recip
    tl.store(output_ptr + offs, vals, mask=mask)

def replacement_func():
    return kernel_wrapper