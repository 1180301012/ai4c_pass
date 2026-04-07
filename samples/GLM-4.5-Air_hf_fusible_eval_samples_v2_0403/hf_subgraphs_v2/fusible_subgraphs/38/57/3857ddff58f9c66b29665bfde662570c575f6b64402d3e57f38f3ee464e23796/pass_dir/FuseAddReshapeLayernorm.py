import torch
import triton
import triton.language as tl

@triton.jit
def simple_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # Mask to ensure we don't go out of bounds
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Calculate
    out = x + y
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

@triton.jit
def fused_add_reshape_layernorm_kernel(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr,
    out_ptr, out_reshaped_ptr,
    n_batch, n_seq, hidden_size,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one hidden dimension position across all batch and sequence
    pid = tl.program_id(0)
    
    # Calculate the offset for this hidden dimension
    hidden_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid hidden dimensions
    hidden_mask = hidden_offset < hidden_size
    
    # Process all batch and sequence elements for this hidden dimension
    for batch_idx in range(n_batch):
        for seq_idx in range(n_seq):
            # Calculate the base offset for this batch and sequence
            base_offset = (batch_idx * n_seq + seq_idx) * hidden_size
            
            # Load input tensors for this hidden dimension
            x_vals = tl.load(x_ptr + base_offset + hidden_offset, mask=hidden_mask, other=0.0)
            y_vals = tl.load(y_ptr + base_offset + hidden_offset, mask=hidden_mask, other=0.0)
            
            # Addition
            sum_vals = x_vals + y_vals
            
            # Store the sum (this is the reshaped output)
            tl.store(out_reshaped_ptr + base_offset + hidden_offset, sum_vals, mask=hidden_mask)
            
            # Load layernorm parameters
            weight = tl.load(weight_ptr + hidden_offset, mask=hidden_mask, other=1.0)
            bias = tl.load(bias_ptr + hidden_offset, mask=hidden_mask, other=0.0)
            
            # Layer normalization computation
            # Calculate mean
            mean = tl.sum(sum_vals) / tl.sum(hidden_mask)
            
            # Calculate variance
            diff = sum_vals - mean
            var = tl.sum(diff * diff) / tl.sum(hidden_mask)
            
            # Normalize and scale
            inv_std = 1.0 / tl.sqrt(var + eps)
            normalized = diff * inv_std
            
            # Apply weight and bias
            out = normalized * weight + bias
            
            # Store the layernorm result
            tl.store(out_ptr + base_offset + hidden_offset, out, mask=hidden_mask)

@torch.fx.wrap
def fused_add_reshape_layernorm(x, y, weight, bias, hidden_size, eps=1e-05):
    # Get input shapes
    batch_size, seq_len, _ = x.shape
    
    # Prepare output tensors
    out_reshaped = torch.empty_like(x)  # Reshaped sum
    out_layernorm = torch.empty_like(x)  # Layer norm result
    
    # Launch kernel
    BLOCK_SIZE = min(1024, triton.next_power_of_2(hidden_size))
    grid = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_reshape_layernorm_kernel[grid](
        x, y,
        weight, bias,
        out_layernorm, out_reshaped,
        batch_size, seq_len, hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_reshaped, out_layernorm

def pattern(x, y):
    # Exactly match the reference example
    return x + y,

def replacement_args(x, y):
    # For simple add pattern, we only need the two input tensors
    return (x, y)

def replacement_func():
    return simple_triton_add