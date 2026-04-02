import torch
import triton
import triton.language as tl

def pattern(in_2):
    # Original computation: transpose last two dimensions
    tmp_4 = in_2.transpose(-2, -1)
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    n,
    c,
    h,
    w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    total_elements = n * c * h * w
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Since we're transposing h and w dimensions, we need to map positions
    # Original layout: [n, c, h, w] -> Output layout: [n, c, w, h]
    if mask[0]:
        # Flatten the tensor index
        linear_idx = offsets
        n_idx = linear_idx // (c * h * w)
        residual = linear_idx % (c * h * w)
        c_idx = residual // (h * w)
        residual = residual % (h * w)
        h_idx_orig = residual // w
        w_idx_orig = residual % w
        
        # Transposed positions: h and w are swapped
        h_idx_new = w_idx_orig
        w_idx_new = h_idx_orig
        
        # Calculate new linear index
        new_linear_idx = (n_idx * (c * h * w) + 
                         c_idx * (h * w) + 
                         h_idx_new * w + w_idx_new)
        
        # Load from original position, store at transposed position
        value = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + new_linear_idx, value, mask=mask)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    # Input shape: [n, c, h, w]
    n, c, h, w = input_tensor.shape
    
    # Output shape after transpose(-2, -1): [n, c, w, h]
    output_shape = (n, c, w, h)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # For transpose, we can optimize by using a more direct approach
    # Actually, let's use a simpler and more efficient method for 2D transpose
    if len(input_tensor.shape) == 4:
        # Handle 4D tensor transpose of last two dimensions
        # We can process the tensor in tiles for better performance
        total_elements = n * c * h * w
        BLOCK_SIZE = 1024
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch transpose kernel
        optimized_transpose_kernel[(num_programs,)](
            input_ptr=input_tensor,
            output_ptr=output,
            n=n,
            c=c,
            h=h,
            w=w,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Fallback for other cases
        output = input_tensor.transpose(-2, -1)
    
    return output

def replacement_func():
    return optimized_transpose