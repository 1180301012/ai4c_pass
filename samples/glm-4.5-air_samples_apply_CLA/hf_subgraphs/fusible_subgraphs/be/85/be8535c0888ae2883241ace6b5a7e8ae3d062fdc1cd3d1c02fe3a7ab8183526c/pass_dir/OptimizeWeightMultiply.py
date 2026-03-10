import torch
import triton
import triton.language as tl

def pattern(weights, normalized):
    """Pattern: Weight multiplication with broadcasting
    Multiplies weight vector [hidden] with normalized tensor [batch, seq, hidden]
    """
    return weights * normalized

def replacement_args(weights, normalized):
    """Extract arguments: weight tensor and normalized tensor"""
    return (weights, normalized)

@triton.jit
def weight_multiply_kernel(
    weights_ptr,
    normalized_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for element-wise multiplication with broadcasting weights"""
    # Each program handles one hidden dimension
    hidden_idx = tl.program_id(0)
    
    # Create stride vectors
    batch_stride = n_seq * n_hidden
    seq_stride = n_hidden
    
    # Load weight for this hidden position (only the first element, since weights is [n_hidden])
    weight = tl.load(weights_ptr + hidden_idx)
    
    # Compute thread range
    hidden_offset = hidden_idx * BLOCK_SIZE
    hidden_mask = hidden_offset + tl.arange(0, BLOCK_SIZE) < n_hidden
    
    # Process each batch and sequence combination
    for batch_idx in range(n_batch):
        for seq_idx in range(n_seq):
            # Compute pointer for current position
            norm_ptr = normalized_ptr + batch_idx * batch_stride + seq_idx * seq_stride + hidden_offset
            out_ptr_offset = out_ptr + batch_idx * batch_stride + seq_idx * seq_stride + hidden_offset
            
            # Load normalized values for this hidden position
            normalized_vals = tl.load(norm_ptr, mask=hidden_mask, other=0.0)
            
            # Apply weight multiplication
            result = normalized_vals * weight
            
            # Store result
            tl.store(out_ptr_offset, result, mask=hidden_mask)

@torch.fx.wrap
def optimized_weight_multiply(weights, normalized):
    """High-performance weight multiplication with broadcasting"""
    if weights.dim() != 1:
        raise ValueError("Expected 1D weights tensor")
    if normalized.dim() != 3:
        raise ValueError("Expected 3D normalized tensor")
    
    n_hidden = weights.shape[0]
    n_batch, n_seq, _ = normalized.shape
    
    if n_hidden != normalized.shape[2]:
        raise ValueError("Hidden dimension mismatch between weights and normalized tensor")
    
    # Set optimal block size
    BLOCK_SIZE = 128
    
    # Calculate grid size
    hidden_programs = (n_hidden + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(normalized)
    
    # Launch kernel
    weight_multiply_kernel[(hidden_programs,)](
        weights_ptr=weights,
        normalized_ptr=normalized,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq,
        n_hidden=n_hidden,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized weight multiplication function"""
    return optimized_weight_multiply