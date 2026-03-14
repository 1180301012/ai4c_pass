import torch
import triton
import triton.language as tl

def pattern(in_4, tmp_2):
    """
    Pattern matching for add + type conversion fusion:
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    """
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    return tmp_4

@triton.jit
def fused_add_type_kernel(
    in4_ptr,       # in_4 pointer
    input_ptr,     # tmp_2 pointer (input from previous operation)
    out_ptr,       # output pointer
    n_sequences,
    n_heads,
    seq_len,
    last_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID mapping
    pid = tl.program_id(0)
    
    # Calculate total number of elements per sequence/position
    elements_per_position = last_dim
    positions_per_program = BLOCK_SIZE // elements_per_position
    total_positions = n_sequences * n_heads * seq_len
    
    # Calculate position indices
    position_idx = pid * positions_per_program
    element_start = position_idx * elements_per_position
    
    # Handle remaining elements
    if position_idx < total_positions:
        end_position = min((pid + 1) * positions_per_program, total_positions)
        
        for pos in range(position_idx, end_position):
            # Calculate indices for this position
            seq_idx = pos // (n_heads * seq_len)
            head_idx = (pos % (n_heads * seq_len)) // seq_len
            elem_idx = pos % seq_len
            
            mask = tl.arange(0, last_dim) < last_dim
            
            # Load in_4 values
            in4_offset = seq_idx * n_heads * seq_len * last_dim + head_idx * seq_len * last_dim + elem_idx * last_dim
            in4_vals = tl.load(in4_ptr + in4_offset, mask=mask, other=0.0)
            
            # Load input values (tmp_2)
            input_offset = seq_idx * n_heads * seq_len * last_dim + head_idx * seq_len * last_dim + elem_idx * last_dim
            input_vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
            
            # Perform addition and convert to float32
            out_vals = tl.cast(in4_vals + input_vals, tl.float32)
            
            # Store result
            out_offset = seq_idx * n_heads * seq_len * last_dim + head_idx * seq_len * last_dim + elem_idx * last_dim
            tl.store(out_ptr + out_offset, out_vals, mask=mask)

@torch.fx.wrap
def fused_add_type(in4, input_val):
    """
    Fused kernel for addition + type conversion
    in4: [n_sequences, n_heads, seq_len, last_dim]
    input_val: [n_sequences, n_heads, seq_len, last_dim]  
    output: [n_sequences, n_heads, seq_len, last_dim] as float32
    """
    # Get tensor shapes
    n_sequences, n_heads, seq_len, last_dim = in4.shape
    
    # Create output tensor (float32)
    out = torch.empty(n_sequences, n_heads, seq_len, last_dim, dtype=torch.float32, device=in4.device)
    
    # Launch kernel
    total_positions = n_sequences * n_heads * seq_len
    BLOCK_SIZE = 1024  # Elements per program (should be divisible by last_dim)
    
    grid = lambda meta: (
        (total_positions + BLOCK_SIZE//last_dim - 1) // (BLOCK_SIZE//last_dim),
        1, 1
    )
    
    fused_add_type_kernel[grid](
        in4_ptr=in4,
        input_ptr=input_val,
        out_ptr=out,
        n_sequences=n_sequences,
        n_heads=n_heads,
        seq_len=seq_len,
        last_dim=last_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_args(in_4, tmp_2):
    return (in_4, tmp_2)

def replacement_func():
    return fused_add_type