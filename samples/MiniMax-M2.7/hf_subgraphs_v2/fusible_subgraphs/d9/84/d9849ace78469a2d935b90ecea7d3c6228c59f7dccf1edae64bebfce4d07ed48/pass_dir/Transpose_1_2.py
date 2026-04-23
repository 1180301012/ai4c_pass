import torch
import triton
import triton.language as tl

# Pattern matching function - match transpose(1, 2)
def pattern(x):
    """
    Match the pattern: transpose(x, 1, 2)
    """
    return x.transpose(1, 2)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for transpose(1, 2)
@triton.jit
def transpose_1_2_kernel(
    x_ptr,
    out_ptr,
    BATCH_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    FEAT_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose kernel that transposes dimensions 1 and 2.
    Input: [BATCH, SEQ, FEAT] -> Output: [BATCH, FEAT, SEQ]
    """
    program_id = tl.program_id(0)
    
    # Output is [BATCH, FEAT, SEQ] with FEAT=128, SEQ=19
    batch_idx = program_id // (FEAT_DIM * SEQ_LEN)
    remaining = program_id % (FEAT_DIM * SEQ_LEN)
    feat_idx = remaining // SEQ_LEN
    seq_idx = remaining % SEQ_LEN
    
    # Input is [BATCH, SEQ, FEAT]
    # Read from [batch_idx, seq_idx, feat_idx]
    input_offset = batch_idx * SEQ_LEN * FEAT_DIM + seq_idx * FEAT_DIM + feat_idx
    
    x_val = tl.load(x_ptr + input_offset)
    
    # Store to output [batch_idx, feat_idx, seq_idx]
    out_offset = program_id
    tl.store(out_ptr + out_offset, x_val)


@torch.fx.wrap
def transpose_1_2_wrapper(x):
    """
    Wrapper function for optimized transpose(1, 2) operation.
    Transposes from [BATCH, SEQ, FEAT] to [BATCH, FEAT, SEQ]
    """
    BATCH_SIZE = x.shape[0]
    SEQ_LEN = x.shape[1]
    FEAT_DIM = x.shape[2]
    
    total_elements = BATCH_SIZE * FEAT_DIM * SEQ_LEN
    
    out = torch.empty([BATCH_SIZE, FEAT_DIM, SEQ_LEN], 
                      dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    num_programs = total_elements
    
    transpose_1_2_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        BATCH_SIZE=BATCH_SIZE,
        SEQ_LEN=SEQ_LEN,
        FEAT_DIM=FEAT_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return transpose_1_2_wrapper