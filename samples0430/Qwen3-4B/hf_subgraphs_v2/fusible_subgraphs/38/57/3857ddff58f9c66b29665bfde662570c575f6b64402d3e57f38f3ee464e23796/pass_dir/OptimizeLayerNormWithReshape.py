import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, in_1, in_0):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 768)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    return tmp_3, tmp_4

def replacement_args(in_2, in_3, in_1, in_0):
    return (in_2, in_3, in_1, in_0)

@triton.jit
def optimized_kernel(
    in_2_ptr,
    in_3_ptr,
    in_1_ptr,
    in_0_ptr,
    out_3_ptr,
    out_4_ptr,
    seq_len: tl.int32,
    hidden_dim: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_start = row * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, hidden_dim)
    
    # Load inputs for current block
    in_2 = tl.zeros((seq_len, block_end - block_start), dtype=tl.float16)
    in_3 = tl.zeros((seq_len, block_end - block_start), dtype=tl.float16)
    
    for h in tl.arange(block_start, block_end):
        for s in tl.arange(seq_len):
            in_2_val = tl.load(in_2_ptr + (s * hidden_dim) + h)
            in_3_val = tl.load(in_3_ptr + (s * hidden_dim) + h)
            in_2[s, h - block_start] = in_2_val
            in_3[s, h - block_start] = in_3_val
    
    # Sum and process layer norm
    in_2_sum = in_2 + in_3
    
    # Layer norm calculation (simplified for this example)
    # For real implementation, use mean/var calculations
    mean = tl.reduce(in_2_sum, axis=0, init=0.0) / seq_len
    var = tl.reduce((in_2_sum - mean) ** 2, axis=0, init=0.0) / seq_len
    
    normalized = (in_2_sum - mean) / tl.sqrt(var + 1e-5)
    out_3 = normalized * tl.load(in_1_ptr + h) + tl.load(in_0_ptr + h)
    
    # Store results
    for s in tl.arange(seq_len):
        for h in tl.arange(block_end - block_start):
            tl.store(out_3_ptr + (s * hidden_dim + h), out_3[s, h])
            tl.store(out_4_ptr + (s * hidden_dim + h), out_3[s, h])

@torch.fx.wrap
def kernel_wrapper(in_2, in_3, in_1, in_0):
    seq_len = in_2.shape[1]
    hidden_dim = in_2.shape[2]
    out_3 = torch.empty((seq_len, hidden_dim), dtype=in_2.dtype)
    out_4 = torch.empty((seq_len, hidden_dim), dtype=in_2.dtype)
    
    BLOCK_SIZE = 128
    num_blocks = (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_kernel[(num_blocks,)](
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_3_ptr=out_3,
        out_4_ptr=out_4,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_3, out_4

def replacement_func():
    return kernel_wrapper