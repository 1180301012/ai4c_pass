import torch
import triton
import triton.language as tl

def pattern(in_2, in_1):
    # Try to match the einsum operation with exactly the same parameter order
    result = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    return result

def replacement_args(in_2, in_1):
    return (in_2, in_1)

@triton.jit
def optimized_einsum_kernel(
    query_ptr, key_ptr, out_ptr,
    b, h, w, j_out,
    BLOCK_SIZE_J: tl.constexpr,
    C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1) 
    pid_w = tl.program_id(2)
    
    # Calculate flat indices for this [b,h,w] position
    query_idx = pid_b * h * w * C + pid_h * w * C + pid_w * C
    out_idx = pid_b * h * w * j_out + pid_h * w * j_out + pid_w * j_out
    
    # Load the entire query vector for this position
    off_c = tl.arange(0, C)
    query = tl.load(query_ptr + query_idx + off_c)
    
    # Process in tiles for better memory coalescing
    off_j = tl.arange(0, BLOCK_SIZE_J)
    
    for j_base in range(0, j_out, BLOCK_SIZE_J):
        off_j_cur = j_base + off_j
        mask = off_j_cur < j_out
        
        # Load key values for this j position
        key_idx = pid_b * h * w * j_out + pid_h * w * j_out + pid_w * j_out + off_j_cur
        key = tl.load(key_ptr + key_idx, mask=mask)
        
        # Compute dot product: query @ key
        # This corresponds to einsum 'bchw,bchj->bhwj'
        result = tl.sum(query * key, axis=0)
        
        # Store result to output
        tl.store(out_ptr + out_idx + off_j_cur, result, mask=mask)

@torch.fx.wrap 
def optimized_einsum_op(in_2, in_1):
    # Get tensor shapes
    b, h, w, c = in_2.shape  # in_2 is query
    _, _, _, j = in_1.shape   # in_1 is key
    j_out = j
    
    # Prepare output tensor [b, h, w, j]
    output_shape = (b, h, w, j_out)
    output = torch.empty(output_shape, dtype=torch.float32, device='cuda')
    
    # Optimize block size based on j dimension
    BLOCK_SIZE_J = min(256, j_out)
    
    # Launch kernel with grid (b, h, w) and template parameter C
    optimized_einsum_kernel[(b, h, w)](
        in_2, in_1, output,
        b, h, w, j_out,
        BLOCK_SIZE_J,
        c  # This becomes the template parameter C
    )
    
    return output

def replacement_func():
    return optimized_einsum_op