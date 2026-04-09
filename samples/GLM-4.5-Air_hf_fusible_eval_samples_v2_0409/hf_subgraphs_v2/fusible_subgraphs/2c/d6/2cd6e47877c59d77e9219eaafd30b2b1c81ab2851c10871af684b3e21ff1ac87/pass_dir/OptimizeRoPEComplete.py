import torch
import triton
import triton.language as tl

def pattern(x, scale):
    """Pattern matching for combined RoPE operations: slice + negate + stack"""
    # Match: tmp_2 = x[(Ellipsis, slice(1, None, 2))]
    tmp_2 = x[(Ellipsis, slice(1, None, 2))]
    # Match: tmp_3 = -tmp_2
    tmp_3 = -tmp_2
    # Match: tmp_4 = x[(Ellipsis, slice(None, None, 2))]
    tmp_4 = x[(Ellipsis, slice(None, None, 2))]
    # Match: tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    # Match: tmp_6 = tmp_5.reshape(scale)
    tmp_6 = tmp_5.reshape(scale)
    # Return the main result
    return tmp_6

def replacement_args(x, scale):
    return (x, scale)

@triton.jit
def fused_rope_complete_kernel(
    x_ptr,
    output_ptr,
    input_size,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # For each output position, we need interleaved negated odd and even elements
    # Output shape: ..., 2 (stacked), ...
    # So output[i, :, j, k] corresponds to stack([negate(x[i, :, 2j+1, k]), x[i, :, 2j, k]])
    
    # Calculate which element we're processing in the flattened space
    if offsets.shape[0] > 0:
        # For each output element (which is half the size due to stacking)
        element_idx = offsets
        output_elem_idx = element_idx
        
        # Map back to input positions
        # Even position in output takes from input even position
        # Odd position in output takes from input odd position (negated)
        even_output_mask = (output_elem_idx % 2) == 0
        odd_output_mask = (output_elem_idx % 2) == 1
        
        # Even output elements: take from input even positions (2*i)
        even_input_idx = output_elem_idx // 2
        even_output_idx = output_elem_idx[even_output_mask]
        even_input_idx_filtered = even_input_idx[even_output_mask]
        
        # Odd output elements: take from input odd positions (2*i + 1) and negate
        odd_input_idx = output_elem_idx // 2
        odd_input_idx_offset = odd_input_idx * 2 + 1
        odd_output_idx = output_elem_idx[odd_output_mask]
        odd_input_idx_filtered = odd_input_idx_offset[odd_output_mask]
        
        # Process even elements
        if even_output_idx.numel() > 0:
            even_input_mask = even_input_idx_filtered < input_size
            if even_input_mask.any():
                even_data = tl.load(x_ptr + even_input_idx_filtered, mask=even_input_mask, other=0.0)
                tl.store(output_ptr + even_output_idx, even_data, mask=even_output_mask[even_output_idx])
        
        # Process odd elements (with negation)
        if odd_output_idx.numel() > 0:
            odd_input_mask = odd_input_idx_filtered < input_size
            if odd_input_mask.any():
                odd_data = tl.load(x_ptr + odd_input_idx_filtered, mask=odd_input_mask, other=0.0)
                negated_odd_data = -odd_data
                tl.store(output_ptr + odd_output_idx, negated_odd_data, mask=odd_output_mask[odd_output_idx])

@torch.fx.wrap
def optimized_rope_complete(x, scale):
    """Optimized complete RoPE operation: slice + negate + stack + reshape"""
    # Create output tensor with target shape
    output = torch.empty(scale, dtype=x.dtype, device=x.device)
    
    # Input and output sizes
    input_size = x.numel()
    output_size = output.numel()
    
    # Block size and grid configuration
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_rope_complete_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        input_size=input_size,
        output_size=output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_rope_complete