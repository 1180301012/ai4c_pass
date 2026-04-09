import torch
import triton
import triton.language as tl

def pattern(query, key):
    """
    Pattern matching for einsum('bchw,bchj->bhwj', query, key)
    This is a batched matrix multiplication along the last dimensions
    """
    result = torch.functional.einsum('bchw,bchj->bhwj', query, key)
    return result

def replacement_args(query, key):
    """
    Extract arguments for the replacement function
    """
    return (query, key)

@triton.jit
def einsum_kernel(
    query_ptr, key_ptr, out_ptr,
    batch_size: tl.constexpr, height: tl.constexpr, width: tl.constexpr, feat_dim: tl.constexpr
):
    """
    Optimized Triton kernel for einsum('bchw,bchj->bhwj', query, key)
    
    This kernel computes: out[b,h,w,j] = sum_{c} query[b,c,h,w] * key[b,c,h,j]
    """
    # Get program IDs  
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    # Only process within bounds - avoid chained boolean operators
    if b >= batch_size:
        return
    if h >= height:
        return
    if w >= width:
        return
    
    # Compute base pointer for this (b, h, w) position in query
    # query[b,c,h,w] -> offset = b*(feat_dim*height*width) + c*(height*width) + h*width + w
    query_offset = b * (feat_dim * height * width) + h * (feat_dim * width) + w * feat_dim
    
    # Compute base pointer for this (b, h) position in key  
    # key[b,c,h,j] -> offset = b*(feat_dim*height*width) + c*(height*width) + h*width + j
    key_offset = b * (feat_dim * height * width) + h * (feat_dim * width)
    
    # Compute base pointer for this (b, h, w) position in output
    # out[b,h,w,j] -> offset = b*(height*width*width) + h*(width*width) + w*width + j
    out_offset = b * (height * width * width) + h * (width * width) + w * width
    
    # Process each j position in the output
    for j in range(width):
        # Compute dot product: sum_{c} query[b,c,h,w] * key[b,c,h,j]
        acc = 0.0
        
        # Load elements in chunks for better memory access
        for c_chunk in range(0, feat_dim, 4):
            # Process up to 4 elements at a time for better performance
            end_c = min(c_chunk + 4, feat_dim)
            
            # Load query and key elements directly and accumulate product
            for c_offset in range(0, end_c - c_chunk):
                query_idx = query_offset + c_chunk + c_offset
                key_idx = key_offset + (c_chunk + c_offset) * width + j
                
                if c_chunk + c_offset < feat_dim:
                    query_val = tl.load(query_ptr + query_idx)
                    key_val = tl.load(key_ptr + key_idx)
                    acc += query_val * key_val
        
        # Store result for this (b, h, w, j)
        out_idx = out_offset + j
        tl.store(out_ptr + out_idx, acc)

@torch.fx.wrap
def optimized_einsum_bchw_bchj_bhwj(query, key):
    """
    Optimized einsum('bchw,bchj->bhwj', query, key) using Triton
    """
    batch_size, height, width, feat_dim = query.shape
    
    # Create output tensor with shape [batch_size, height, width, width]
    out = torch.empty((batch_size, height, width, width), dtype=query.dtype, device=query.device)
    
    # Launch kernel - one program per (batch, height, width) combination
    grid = (batch_size, height, width)
    
    # Launch kernel
    einsum_kernel[grid](
        query, key, out,
        batch_size, height, width, feat_dim
    )
    
    return out

def replacement_func():
    """
    Returns the optimized function
    """
    return optimized_einsum_bchw_bchj_bhwj