import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    """Pattern: Linear + Permute (0,2,1) fusion"""
    linear = torch.nn.functional.linear(x, weight, bias)
    result = linear.permute(0, 2, 1)
    return result

def replacement_args(x, weight, bias):
    """Extract arguments for the fused kernel"""
    return (x, weight, bias)

@triton.jit
def linear_permute_fused_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size: tl.constexpr, seq_len: tl.constexpr, hidden_in: tl.constexpr, hidden_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized fused linear + permute kernel with 3D grid"""
    # Program IDs for 3D grid
    batch_id = tl.program_id(0)
    hidden_id = tl.program_id(1)
    seq_block_id = tl.program_id(2)
    
    # Calculate base sequence index for this block
    seq_base = seq_block_id * BLOCK_SIZE
    
    # Load weights for this hidden dimension (same for all sequence positions in block)
    w_block = tl.load(weight_ptr + hidden_id * hidden_in + tl.arange(0, hidden_in), 
                      mask=tl.arange(0, hidden_in) < hidden_in, other=0.0)  # [hidden_in]
    
    # Load bias for this hidden dimension
    bias_val = tl.load(bias_ptr + hidden_id, mask=hidden_id < hidden_out, other=0.0)  # scalar
    
    # Process multiple sequence positions in the block
    for i in range(BLOCK_SIZE):
        seq_idx = seq_base + i
        
        # Skip if sequence index is out of bounds
        if seq_idx < seq_len:
            # Load x for this specific batch and sequence position: [hidden_in]
            x_offset = (batch_id * seq_len + seq_idx) * hidden_in
            x_val = tl.load(x_ptr + x_offset + tl.arange(0, hidden_in), 
                           mask=tl.arange(0, hidden_in) < hidden_in, other=0.0)
            
            # Compute dot product using tl.sum (for 1D vectors)
            result = tl.sum(x_val * w_block) + bias_val
            
            # Store result with permuted layout: [batch_size, hidden_out, seq_len]
            out_offset = (batch_id * hidden_out + hidden_id) * seq_len + seq_idx
            tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def linear_permute_optimized(x, weight, bias):
    """Optimized linear + permute fusion"""
    # Get input dimensions
    batch_size, seq_len, hidden_in = x.shape
    hidden_out = weight.shape[0]
    
    # Output shape after permutation: [batch_size, hidden_out, seq_len]
    output = torch.empty((batch_size, hidden_out, seq_len), dtype=x.dtype, device=x.device)
    
    # Conservative block size to balance overhead and memory efficiency
    BLOCK_SIZE = 32  # Process 32 sequence positions per thread - balanced approach
    
    # Calculate grid dimensions - using 3D grid
    grid_batch = batch_size
    grid_hidden = hidden_out
    grid_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with 3D grid
    linear_permute_fused_kernel[(grid_batch, grid_hidden, grid_seq)](
        x_ptr=x, weight_ptr=weight, bias_ptr=bias, out_ptr=output,
        batch_size=batch_size, seq_len=seq_len, hidden_in=hidden_in, hidden_out=hidden_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return linear_permute_optimized