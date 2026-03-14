import torch
import triton
import triton.language as tl

# Pattern matching function - matches view + unsqueeze pattern
def pattern(relu_out, shape_0, shape_1, shape_2):
    """
    Match pattern: view -> unsqueeze
    relu_out: output of relu
    shape_0, shape_1, shape_2: the target view shape
    """
    view_out = relu_out.view(shape_0, shape_1, shape_2)
    unsqueeze_out = view_out.unsqueeze(1)
    return unsqueeze_out

# Argument extraction function
def replacement_args(relu_out, shape_0, shape_1, shape_2):
    return (relu_out, shape_0, shape_1, shape_2)

# Optimized Triton kernel with autotuning for better performance
@triton.autotune(
    configs=[
        # Different block sizes for different tensor sizes
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_view_unsqueeze_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store output
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_view_unsqueeze_kernel_wrapper(input_tensor, shape_0, shape_1, shape_2):
    """
    Fused view + unsqueeze operation using optimized Triton kernel.
    Uses reshape to achieve zero-copy when possible.
    """
    # Total elements must match
    expected_elements = shape_0 * shape_1 * shape_2
    actual_elements = input_tensor.numel()
    
    if actual_elements == expected_elements:
        # Direct reshape to final shape: [shape_0, 1, shape_1, shape_2]
        # This is zero-copy when contiguous
        output_shape = (shape_0, 1, shape_1, shape_2)
        output = input_tensor.reshape(output_shape)
        return output
    else:
        # Fallback: use Triton kernel for non-contiguous case
        input_flat = input_tensor.view(-1)
        n_elements = input_flat.numel()
        
        output_shape = (shape_0, 1, shape_1, shape_2)
        output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
        output_flat = output.view(-1)
        
        # Launch kernel
        num_programs = (n_elements + 256 - 1) // 256
        fused_view_unsqueeze_kernel[(num_programs,)](
            input_ptr=input_flat,
            output_ptr=output_flat,
            n_elements=n_elements,
            BLOCK_SIZE=256,
        )
        
        return output

def replacement_func():
    return fused_view_unsqueeze_kernel_wrapper