import torch
import triton
import triton.language as tl

# Pattern matching function (exact sequence of operations)
def pattern(in_3, in_1, in_5):
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_4 = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 6, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    return tmp_8

# Argument extraction function
def replacement_args(in_3, in_1, in_5):
    return (in_3, in_1, in_5)

# Triton kernel for optimized rotary embedding
@triton.jit
def rotary_kernel(
    in_3_ptr, in_1_ptr, in_5_ptr,
    out_ptr,
    seq_len, head_dim,
    batch_size, num_heads,
    BLOCK_SIZE: tl.constexpr
):
    # Compute global thread index
    # Each thread processes one element
    pid = tl.program_id(0)
    tid = tl.thread_id(0)
    n_elements = batch_size * num_heads * seq_len * head_dim
    idx = pid * BLOCK_SIZE + tid
    if idx >= n_elements:
        return
    
    # Calculate 4D indices from linear index
    head_dim_idx = idx % head_dim
    seq_idx = (idx // head_dim) % seq_len
    head_idx = (idx // (head_dim * seq_len)) % num_heads
    batch_idx = (idx // (head_dim * seq_len * num_heads)) % batch_size
    
    # Determine parity (even/odd) for head_dim
    even = (head_dim_idx % 2) == 0
    factor = 1 if even else -1
    
    # Load input values
    in_3_val = tl.load(in_3_ptr + idx)
    in_1_val = tl.load(in_1_ptr + seq_idx * head_dim + head_dim_idx)
    in_5_val = tl.load(in_5_ptr + seq_idx * head_dim + head_dim_idx)
    
    # Compute optimized value
    out_val = in_3_val * (in_1_val + factor * in_5_val)
    
    # Store result
    tl.store(out_ptr + idx, out_val)

# Kernel wrapper (must be wrapped by @torch.fx.wrap)
@torch.fx.wrap
def rotary_embedding(in_3, in_1, in_5):
    batch_size, num_heads, seq_len, head_dim = in_3.shape
    N = batch_size * num_heads * seq_len * head_dim
    
    # Create output tensor
    out = torch.empty_like(in_3)
    
    # Block size optimized for NVIDIA GPUs (power of 2 for efficiency)
    BLOCK_SIZE = 256
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    rotary_kernel[(num_blocks,)](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_5_ptr=in_5,
        out_ptr=out,
        seq_len=seq_len,
        head_dim=head_dim,
        batch_size=batch_size,
        num_heads=num_heads,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (returns callable)
def replacement_func():
    return rotary_embedding