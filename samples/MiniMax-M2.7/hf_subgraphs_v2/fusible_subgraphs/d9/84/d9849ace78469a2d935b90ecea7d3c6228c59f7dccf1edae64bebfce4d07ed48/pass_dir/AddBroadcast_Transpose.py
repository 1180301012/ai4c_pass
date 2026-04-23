import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    """
    Match the pattern: transpose(x, 1, 2)
    """
    return x.transpose(1, 2)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel that fuses add with broadcasting and transpose
@triton.jit
def fused_add_broadcast_transpose_kernel(
    x_ptr,          # in_0: [128, 1]
    y_ptr,          # in_1: [1, 128, 19]
    out_ptr,        # output: [1, 19, 128]
    BATCH_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    FEAT_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that:
    1. Adds x [128, 1] to y [1, 128, 19] with broadcasting
    2. Transposes result from [1, 128, 19] to [1, 19, 128]
    
    The output dimensions are [1, 19, 128]
    - BATCH_SIZE = 1
    - SEQ_LEN = 19 (after transpose from original dim 2)
    - FEAT_DIM = 128 (after transpose from original dim 1)
    
    Original input dimensions:
    - in_0: [128, 1] -> we access [feat_dim, 0]
    - in_1: [1, 128, 19] -> we access [0, feat_dim, seq_dim]
    
    Output [1, 19, 128]: we store to [0, seq_dim, feat_dim]
    where seq_dim maps from original y's dim 2
    and feat_dim maps from original y's dim 1
    """
    # Program id for output elements
    program_id = tl.program_id(0)
    
    # Output is [1, 19, 128] -> 1 * 19 * 128 = 2432 total elements
    # Calculate output indices
    # Flattened output index: program_id
    # Output shape: [1, 19, 128] -> [BATCH=1, SEQ_LEN=19, FEAT=128]
    batch_idx = 0  # Only batch 0
    seq_idx = program_id // 128  # 0-18
    feat_idx = program_id % 128   # 0-127
    
    # For broadcasting: in_0 [128, 1] is accessed at [feat_idx, 0]
    # in_1 [1, 128, 19] is accessed at [0, feat_idx, seq_idx]
    x_offset = feat_idx  # [128, 1] -> linear index
    y_offset = feat_idx * 19 + seq_idx  # [1, 128, 19] -> [0, feat, seq]
    
    # Load values
    x_val = tl.load(x_ptr + x_offset)
    y_val = tl.load(y_ptr + y_offset)
    
    # Add with broadcasting (broadcasting happens in the access pattern)
    result = x_val + y_val
    
    # Store to transposed output [1, 19, 128]
    out_offset = program_id
    tl.store(out_ptr + out_offset, result)


@torch.fx.wrap
def fused_add_broadcast_transpose_wrapper(in_0, in_1):
    """
    Wrapper function for the fused add+broadcast+transpose operation.
    Performs: transpose(in_1 + in_0, 1, 2)
    """
    # Input shapes: in_0 [128, 1], in_1 [1, 128, 19]
    # Output shape: [1, 19, 128]
    
    # Get dimensions
    batch_size = 1
    seq_len = in_1.shape[2]  # 19
    feat_dim = in_1.shape[1]  # 128
    
    # Total output elements
    total_elements = batch_size * seq_len * feat_dim  # 1 * 19 * 128 = 2432
    
    # Allocate output
    out = torch.empty([batch_size, seq_len, feat_dim], 
                      dtype=in_1.dtype, device=in_1.device)
    
    # Grid: one program per output element
    BLOCK_SIZE = 128
    num_programs = total_elements
    
    fused_add_broadcast_transpose_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        BATCH_SIZE=batch_size,
        SEQ_LEN=seq_len,
        FEAT_DIM=feat_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_add_broadcast_transpose_wrapper