import torch
import triton
import triton.language as tl

def pattern(x):
    """Match reshape followed by permute"""
    tmp = x.reshape(1, 16, 16, -1)
    result = tmp.permute(0, 3, 1, 2)
    return result

def replacement_args(x):
    """Extract arguments for optimized kernel"""
    return (x,)

@triton.jit
def fused_reshape_permute_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused reshape and permute kernel"""
    # Program identifiers
    batch_id = tl.program_id(0)
    hidden_id = tl.program_id(1)
    
    # Calculate offsets
    offset_batch = batch_id * 256 * hidden_size  # 256 = 16*16
    offset_hidden = hidden_id * BLOCK_SIZE
    
    # Bounds for hidden dimension
    mask_hidden = offset_hidden + tl.arange(0, BLOCK_SIZE) < hidden_size
    
    # Load input data: [batch, 256, hidden]
    offset_base = offset_batch + offset_hidden
    mask_seq = tl.arange(0, 256)[:, None]  # Row vector for broadcasting
    
    x = tl.load(x_ptr + offset_base + mask_seq * hidden_size, mask=mask_seq, other=0.0)
    
    # Reshape to [batch, 16, 16, hidden] and permute to [batch, hidden, 16, 16]
    # Calculate new indices in permuted output
    seq_flat = tl.arange(0, 256)  # 0..255
    seq_row = seq_flat // 16      # Which row (0..15)
    seq_col = seq_flat % 16       # Which column (0..15)
    
    # Reshape and permute: [batch, 256, hidden] -> [batch, hidden, 16, 16]
    # We need to reorder dimensions: from [n, h, s] to [n, h, height, width]
    # where s = height * width = 16 * 16 = 256
    output_flat = seq_row * 16 + seq_col  # Reorder flat sequence to row-major
    
    # Load data in the new order and store directly to output
    offset_out = batch_id * hidden_size * 256 + offset_hidden * 256 + output_flat
    
    # Store result in the permuted order
    tl.store(out_ptr + offset_out, x, mask=mask_seq and mask_hidden[hidden_id])

@torch.fx.wrap
def fused_reshape_permute(x):
    """Optimized fused reshape and permute function"""
    # Get tensor shapes
    n_batch, n_seq, hidden_size = x.shape
    assert n_seq == 256, f"Expected sequence length 256, got {n_seq}"
    
    # Create output tensor: [n_batch, hidden_size, 16, 16]
    out = torch.empty((n_batch, hidden_size, 16, 16), dtype=x.dtype, device=x.device)
    
    # Configure kernel parameters
    BLOCK_SIZE = min(1024, hidden_size)
    hidden_blocks = triton.cdiv(hidden_size, BLOCK_SIZE)
    
    # Launch kernel with 2D grid: (batch_size, hidden_blocks)
    grid = (n_batch, hidden_blocks)
    
    # Launch kernel
    fused_reshape_permute_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_batch=n_batch,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return fused_reshape_permute