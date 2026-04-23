import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,  # Input: (batch*seq, hidden)
    b_ptr,  # Weight: (hidden, hidden)
    bias_ptr,  # Bias: (hidden)
    out_ptr,  # Output: (batch*seq, hidden)
    seq, hidden,  # Sequence length, hidden size
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    # Compute thread indices
    pid_seq = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    seq_start = pid_seq * BLOCK_SEQ
    hidden_start = pid_hidden * BLOCK_HIDDEN
    
    # Shared memory for accumulating
    acc = tl.zeros((BLOCK_SEQ, BLOCK_HIDDEN), dtype=tl.float32)
    
    # Iterate over k (hidden dimension)
    for k in range(0, hidden, BLOCK_HIDDEN):
        # Load a tile from A
        a_tile = tl.load(
            a_ptr + (seq_start * hidden + k),
            shape=(BLOCK_SEQ, min(BLOCK_HIDDEN, hidden - k)),
            mask=tl.arange(0, BLOCK_SEQ)[:, None] < BLOCK_SEQ,
        )
        
        # Load a tile from B
        b_tile = tl.load(
            b_ptr + (k * hidden + hidden_start),
            shape=(min(BLOCK_HIDDEN, hidden - k), BLOCK_HIDDEN),
            mask=tl.arange(0, min(BLOCK_HIDDEN, hidden - k))[:, None] < BLOCK_HIDDEN,
        )
        
        # Multiply and accumulate
        acc += tl.dot(a_tile, b_tile)
        
    # Convert to input dtype (bfloat16)
    c_tile = acc.to(tl.float16)
    
    # Add bias
    bias = tl.load(bias_ptr + hidden_start, shape=(BLOCK_HIDDEN,), mask=tl.arange(0, BLOCK_HIDDEN) < BLOCK_HIDDEN)
    c_tile += bias
    
    # Store result
    tl.store(
        out_ptr + (seq_start * hidden + hidden_start),
        c_tile,
        mask=tl.arange(0, BLOCK_SEQ)[:, None] < BLOCK_SEQ,
    )

@torch.fx.wrap
def optimized_linear(in_3, in_1, in_0):
    batch, seq, hidden = in_3.shape
    in_3_flat = in_3.view(-1, hidden)
    out_flat = torch.empty((batch * seq, hidden), dtype=in_3_flat.dtype)
    
    BLOCK_SEQ = 32
    BLOCK_HIDDEN = 32
    grid_seq = (batch * seq + BLOCK_SEQ - 1) // BLOCK_SEQ
    grid_hidden = (hidden + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN
    
    matmul_kernel[(grid_seq, grid_hidden)](
        in_3_flat, in_1, in_0, out_flat, seq, hidden, BLOCK_SEQ, BLOCK_HIDDEN
    )
    
    return out_flat.view(batch, seq, hidden)

def pattern(in_3, in_1, in_0):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    return linear

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

def replacement_func():
    return optimized_linear