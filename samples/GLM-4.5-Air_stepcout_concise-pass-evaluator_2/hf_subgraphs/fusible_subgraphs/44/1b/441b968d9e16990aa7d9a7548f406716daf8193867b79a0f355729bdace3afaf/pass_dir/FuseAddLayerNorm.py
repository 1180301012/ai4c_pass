import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, y):
    """Match addition followed by layer normalization"""
    tmp = y + x
    result = torch.nn.functional.layer_norm(tmp, (512,), weight, bias, 1e-05)
    return result

def replacement_args(x, weight, bias, y):
    """Extract arguments for optimized kernel"""
    return (x, weight, bias, y)

@triton.jit
def fused_add_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_batch,
    n_seq,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused addition and layer normalization kernel"""
    # Program identifiers
    batch_id = tl.program_id(0)
    hidden_id = tl.program_id(1)
    
    # Calculate offsets
    offset_batch = batch_id * n_seq * hidden_size
    offset_hidden = hidden_id * BLOCK_SIZE_N
    
    # Bounds for hidden dimension
    mask_hidden = offset_hidden + tl.arange(0, BLOCK_SIZE_N) < hidden_size
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offset_hidden, mask=mask_hidden, other=0.0)
    bias = tl.load(bias_ptr + offset_hidden, mask=mask_hidden, other=0.0)
    
    # Load input data for this batch and hidden position across all sequence positions
    offset_base = offset_batch + offset_hidden
    mask_seq = tl.arange(0, n_seq)[:, None]  # Row vector for broadcasting
    
    x = tl.load(x_ptr + offset_base + mask_seq * hidden_size, mask=mask_seq, other=0.0)
    y = tl.load(y_ptr + offset_base + mask_seq * hidden_size, mask=mask_seq, other=0.0)
    
    # Fused computation: addition + layer normalization
    out = y + x
    
    # Apply layer normalization across sequence dimension
    mean = tl.sum(out, axis=0) / n_seq
    variance = tl.sum((out - mean) * (out - mean), axis=0) / n_seq
    normalized = (out - mean) / tl.sqrt(variance + eps)
    
    # Apply weight and bias
    result = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offset_base + mask_seq * hidden_size, result, mask=mask_seq)

@torch.fx.wrap
def fused_add_layer_norm(x, weight, bias, y):
    """Optimized fused addition and layer normalization function"""
    # Get tensor shapes
    n_batch, n_seq, hidden_size = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Configure kernel parameters
    BLOCK_SIZE_N = min(1024, hidden_size)
    hidden_blocks = triton.cdiv(hidden_size, BLOCK_SIZE_N)
    
    # Launch kernel with 2D grid: (batch_size, hidden_blocks)
    grid = (n_batch, hidden_blocks)
    
    # Launch kernel
    fused_add_layer_norm_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq,
        hidden_size=hidden_size,
        eps=1e-05,
        BLOCK_SIZE_M=BLOCK_SIZE_N,  # Keep this parameter for interface compatibility
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return fused_add_layer_norm