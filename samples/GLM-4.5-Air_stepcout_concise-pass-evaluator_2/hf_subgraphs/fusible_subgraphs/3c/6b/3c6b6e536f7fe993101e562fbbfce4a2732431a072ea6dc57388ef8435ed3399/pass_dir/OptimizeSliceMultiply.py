import torch
import triton
import triton.language as tl

def pattern(in_0, in_3):
    """
    Pattern matching for slice + multiply operation:
    tmp_5 = in_0[slice(None, None, None), slice(None, None, None), slice(None, seq_limit, None), slice(None, None, None)]
    tmp_7 = in_3 * tmp_5
    """
    tmp_5 = in_0[slice(None, None, None), slice(None, None, None), slice(None, in_0.shape[2], None), slice(None, None, None)]
    tmp_7 = in_3 * tmp_5
    return tmp_7

@triton.jit
def slice_multiply_kernel(
    in0_ptr,       # in_0 pointer
    in3_ptr,       # in_3 pointer
    out_ptr,       # output pointer
    n_sequences0,
    n_heads0,
    seq_len0,
    last_dim0,
    
    n_sequences3,
    n_heads3,
    seq_len3,
    last_dim3,
    
    # Slice parameters 
    seq_limit,     # The sequence limit for slicing
    
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID mapping
    pid = tl.program_id(0)
    
    # Note: We assume that in_0 and in_3 have compatible shapes after slicing
    # Based on the patterns, in_0 is sliced along sequence dimension to match in_3
    
    # Calculate indices
    elem_per_program = BLOCK_SIZE // last_dim3
    total_elem = n_sequences3 * n_heads3 * seq_len3
    elem_idx = pid * elem_per_program
    
    # Load multiple elements per program for better efficiency
    if elem_idx < total_elem:
        end_elem = min((pid + 1) * elem_per_program, total_elem)
        
        for elem in range(elem_idx, end_elem):
            # Calculate indices for this element
            seq_idx = elem // (n_heads3 * seq_len3)
            head_idx = (elem % (n_heads3 * seq_len3)) // seq_len3
            pos_idx = elem % seq_len3
            
            # Check if position is within slice limit
            if pos_idx < seq_limit:
                mask = tl.arange(0, last_dim3) < last_dim3
                
                # Load from in_3 (full tensor)
                in3_offset = seq_idx3 * n_heads3 * seq_len3 * last_dim3 + head_idx * seq_len3 * last_dim3 + pos_idx * last_dim3
                in3_vals = tl.load(in3_ptr + in3_offset, mask=mask, other=0.0)
                
                # Load sliced portion from in_0
                in0_offset = seq_idx0 * n_heads0 * seq_len0 * last_dim0 + head_idx * seq_len0 * last_dim0 + pos_idx * last_dim0
                in0_vals = tl.load(in0_ptr + in0_offset, mask=mask, other=0.0)
                
                # Perform multiplication
                out_vals = in3_vals * in0_vals
                
                # Store result
                out_offset = seq_idx3 * n_heads3 * seq_len3 * last_dim3 + head_idx * seq_len3 * last_dim3 + pos_idx * last_dim3
                tl.store(out_ptr + out_offset, out_vals, mask=mask)

@torch.fx.wrap  
def optimized_slice_multiply(in0, in3, seq_limit):
    """
    Optimized kernel for slice from in_0 followed by multiplication with in_3
    in0: [n_sequences0, n_heads0, seq_len0, last_dim0] - source for slice
    in3: [n_sequences3, n_heads3, seq_len3, last_dim3] - multiplier
    seq_limit: sequence dimension limit for slicing
    output: [n_sequences3, n_heads3, seq_len3, last_dim3]
    """
    # Get tensor shapes
    n_sequences0, n_heads0, seq_len0, last_dim0 = in0.shape
    n_sequences3, n_heads3, seq_len3, last_dim3 = in3.shape
    
    # Create output tensor
    out = torch.empty(n_sequences3, n_heads3, seq_len3, last_dim3, dtype=in3.dtype, device=in3.device)
    
    # Launch kernel
    total_elements = n_sequences3 * n_heads3 * seq_len3
    BLOCK_SIZE = 1024  # Elements per program
    
    # Ensure we're using the correct indices for each tensor
    # Based on the pattern, we need to handle the slice properly
    
    # For simplicity, let's use a different approach that handles the slicing more cleanly
    @triton.jit
    def simple_slice_multiply_kernel(
        in0_ptr,
        in3_ptr, 
        out_ptr,
        n_sequences,
        n_heads,
        seq_len,
        last_dim,
        seq_limit,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        stride = tl.num_programs(0)
        
        total_positions = n_sequences * n_heads * seq_len
        positions_per_program = (BLOCK_SIZE + (last_dim - 1)) // last_dim
        pos_idx = pid * positions_per_program
        
        if pos_idx < total_positions:
            end_pos = min((pid + 1) * positions_per_program, total_positions)
            
            for pos in range(pos_idx, end_pos):
                seq_idx = pos // (n_heads * seq_len)
                head_idx = (pos % (n_heads * seq_len)) // seq_len
                elem_idx = pos % seq_len
                
                if elem_idx < seq_limit:  # Only process within slice limit
                    mask = tl.arange(0, last_dim) < last_dim
                    
                    # Load from in_3
                    in3_offset = seq_idx * n_heads * seq_len * last_dim + head_idx * seq_len * last_dim + elem_idx * last_dim
                    in3_vals = tl.load(in3_ptr + in3_offset, mask=mask, other=0.0)
                    
                    # Load from in_0 (sliced implicitly)
                    in0_offset = seq_idx * n_heads * seq_len * last_dim + head_idx * seq_len * last_dim + elem_idx * last_dim
                    in0_vals = tl.load(in0_ptr + in0_offset, mask=mask, other=0.0)
                    
                    # Multiply
                    out_vals = in3_vals * in0_vals
                    
                    # Store
                    out_offset = seq_idx * n_heads * seq_len * last_dim + head_idx * seq_len * last_dim + elem_idx * last_dim
                    tl.store(out_ptr + out_offset, out_vals, mask=mask)
    
    grid = lambda meta: (
        (total_elements + BLOCK_SIZE//last_dim3 - 1) // (BLOCK_SIZE//last_dim3),
        1, 1
    )
    
    simple_slice_multiply_kernel[grid](
        in0_ptr=in0,
        in3_ptr=in3,
        out_ptr=out,
        n_sequences=n_sequences3,
        n_heads=n_heads3,
        seq_len=seq_len3,
        last_dim=last_dim3,
        seq_limit=seq_limit,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_args(in_0, in_3):
    # Extract the sequence limit from in_0's shape (this varies by graph)
    seq_limit = in_0.shape[2]
    return (in_0, in_3, seq_limit)

def replacement_func():
    return optimized_slice_multiply