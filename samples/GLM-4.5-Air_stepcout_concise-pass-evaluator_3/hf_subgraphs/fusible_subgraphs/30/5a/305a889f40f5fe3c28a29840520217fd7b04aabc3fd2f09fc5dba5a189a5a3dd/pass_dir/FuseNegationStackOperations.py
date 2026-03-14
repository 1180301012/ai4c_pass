import torch
import triton
import triton.language as tl

def pattern(neg_input, slice_input):
    """
    Pattern that matches: negate -> slice -> stack
    
    This pattern appears in all graphs with different inputs:
    Graphs 2-4: tmp_1 = -in_3, tmp_2 = in_2[..., slice(None, None, 2)], tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    Graph 1: tmp_1 = -in_1, tmp_2 = in_3[..., slice(None, None, 2)], tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    """
    # Original operations
    tmp_1 = -neg_input
    tmp_2 = slice_input[Ellipsis, slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    return (tmp_3,)

@triton.jit
def fused_neg_slice_stack_kernel(
    neg_ptr,
    slice_ptr,
    out_ptr,
    neg_batch: tl.constexpr,
    neg_heads: tl.constexpr,
    neg_seq: tl.constexpr,
    neg_feat: tl.constexpr,
    slice_batch: tl.constexpr,
    slice_heads: tl.constexpr,
    slice_seq: tl.constexpr,
    slice_feat: tl.constexpr,
    out_batch: tl.constexpr,
    out_heads: tl.constexpr,
    out_seq: tl.constexpr,
    out_feat: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines negation, slicing, and stacking
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate total elements and work per program
    total_elements = out_batch * out_heads * out_seq * out_feat
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate start and end indices for this program
    start_idx = pid * elements_per_program
    end_idx = min(start_idx + elements_per_program, total_elements)
    
    if start_idx >= total_elements:
        return
    
    # Helper function to get multi-dimensional indices
    def get_indices(idx):
        # Unflatten 1D index to 4D indices
        total = idx
        f = total // (out_heads * out_seq * out_feat)
        total = total % (out_heads * out_seq * out_feat)
        h = total // (out_seq * out_feat)
        total = total % (out_seq * out_feat)
        w = total // out_feat
        c = total % out_feat
        return f, h, w, c
    
    # Process elements in this program
    for idx in range(start_idx, end_idx):
        f, h, w, c = get_indices(idx)
        
        # Calculate offset for negation input (half the features)
        neg_c = c // 2
        neg_f, neg_h = f, h  # Same batch and head
        neg_offset = neg_f * neg_heads * neg_seq * neg_feat + \
                     neg_h * neg_seq * neg_feat + \
                     w * neg_feat + neg_c
        neg_val = tl.load(neg_ptr + neg_offset, mask=(neg_f < neg_batch and neg_h < neg_heads and w < neg_seq and neg_c < neg_feat), other=0.0)
        
        # Calculate offset for slice input (every other sequence element)
        slice_w = w * 2 + (c % 2)
        if slice_w < slice_seq:  # Check if within bounds
            slice_offset = f * slice_heads * slice_seq * slice_feat + \
                          h * slice_seq * slice_feat + \
                          slice_w * slice_feat + neg_c
            slice_val = tl.load(slice_ptr + slice_offset, mask=(f < slice_batch and h < slice_heads and slice_w < slice_seq and neg_c < slice_feat), other=0.0)
        else:
            slice_val = 0.0
        
        # Stack: alternate between negation and slice values
        if c % 2 == 0:
            stacked_val = neg_val
        else:
            stacked_val = slice_val
        
        # Store result
        tl.store(out_ptr + idx, stacked_val)

@torch.fx.wrap  
def fused_neg_slice_stack(neg_input, slice_input):
    """Wrapper for the fused negation-slicing-stacking kernel"""
    # Validate input shapes
    if len(neg_input.shape) != 4 or len(slice_input.shape) != 4:
        # Fall back to original computation using only element-wise operations
        tmp_1 = -neg_input
        tmp_2 = slice_input[Ellipsis, slice(None, None, 2)]
        # For fallback, create stacked result manually
        stacked_shape = tmp_2.shape
        output = torch.empty(stacked_shape, dtype=torch.float32, device=neg_input.device)
        # Copy slice values to even positions and neg values to odd positions
        output[..., 0::2] = tmp_2
        output[..., 1::2] = tmp_1
        return output
    
    # Get shapes
    neg_shape = neg_input.shape
    slice_shape = slice_input.shape
    
    # Output shape is same as slice_shape
    out_shape = slice_shape
    
    # Create output tensor
    output = torch.empty(out_shape, dtype=torch.float32, device=neg_input.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    total_elements = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Extract shape dimensions as scalar parameters
    fused_neg_slice_stack_kernel[(num_programs,)](
        neg_input,
        slice_input,
        output,
        neg_shape[0], neg_shape[1], neg_shape[2], neg_shape[3],  # neg batch, heads, seq, feat
        slice_shape[0], slice_shape[1], slice_shape[2], slice_shape[3],  # slice batch, heads, seq, feat
        out_shape[0], out_shape[1], out_shape[2], out_shape[3],  # out batch, heads, seq, feat
        BLOCK_SIZE,
    )
    
    return output

def replacement_args(neg_input, slice_input):
    return (neg_input, slice_input)

def replacement_func():
    return fused_neg_slice_stack