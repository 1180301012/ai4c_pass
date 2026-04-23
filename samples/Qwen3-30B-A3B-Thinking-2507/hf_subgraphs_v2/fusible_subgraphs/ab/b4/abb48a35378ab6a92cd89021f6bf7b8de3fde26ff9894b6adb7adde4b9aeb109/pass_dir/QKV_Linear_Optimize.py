import torch
import triton
import triton.language as tl

# Pattern matching function to match linear + reshape + permute + unbind
# Matches the exact structure in model.py for convit_tiny with reshape (1,197,3,4,48)
def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    reshaped = linear.reshape(1, 197, 3, 9, 48)
    permuted = reshaped.permute(2, 0, 3, 1, 4)
    unbound = permuted.unbind(0)
    return unbound[0], unbound[1].transpose(-2, -1), unbound[2]

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def qkv_linear_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    seq,
    m,
    n,
    d,
    block_size_d: tl.constexpr = 64
):
    # Program IDs: i (0-2), j (0-m-1), seq_idx (0-seq-1)
    i = tl.program_id(0)
    j = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Compute offset for weight (i, j)
    weight_offset = i * (m * n) + j * n
    
    # Input for current sequence
    input_offset = seq_idx * d
    
    # Initialize accumulator for n (48)
    acc = tl.zeros((n,), dtype=tl.float32)
    
    # Process d dimension in blocks
    d_block_start = tl.program_id(3) * block_size_d
    d_block_end = min(d_block_start + block_size_d, d)
    for d_start in range(d_block_start, d_block_end, block_size_d):
        d_mask = (d_start + tl.arange(0, block_size_d)) < d_block_end
        input_val = tl.load(input_ptr + input_offset + d_start, mask=d_mask, other=0.0)
        weight_vals = tl.load(
            weight_ptr + weight_offset + d_start * (m * n),
            mask=d_mask, 
            other=0.0
        )
        acc += weight_vals * input_val

    # Store output [3, 1, m, seq, n] at (i, 0, j, seq_idx)
    output_offset = i * (m * seq * n) + j * (seq * n) + seq_idx * n
    tl.store(output_ptr + output_offset, acc, mask=tl.arange(0, n) < n)

@torch.fx.wrap
def qkv_linear_wrapper(in_0, in_1):
    # Extract shapes
    batch, seq, d = in_1.shape
    k, d_weight = in_0.shape
    assert d == d_weight, "Input and weight dims mismatch"
    m, n = 9, 48
    assert k == 3 * m * n, f"k should be 3*m*n={3*m*n}, got {k}"
    
    # Create output tensor [3, 1, m, seq, n]
    output = torch.empty((3, 1, m, seq, n), dtype=in_0.dtype, device=in_0.device)
    
    # Configure Triton grid
    num_i = 3
    num_j = m
    num_seq = seq
    num_d_blocks = (d + 64 - 1) // 64
    grid = (num_i, num_j, num_seq)
    
    # Launch kernel
    qkv_linear_kernel[grid](
        weight_ptr=in_0,
        input_ptr=in_1,
        output_ptr=output,
        seq=seq,
        m=m,
        n=n,
        d=d,
        block_size_d=64
    )
    
    # Unbind and return with transpose on second element
    unbound = output.unbind(0)
    return unbound[0], unbound[1].transpose(-2, -1), unbound[2]

def replacement_func():
    return qkv_linear_wrapper