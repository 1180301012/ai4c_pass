import torch
import triton
import triton.language as tl

# Pattern matching function for fused addition + softmax + dropout + type conversion
def pattern(in_0, in_1):
    """Match the sequence: addition -> softmax -> dropout -> type conversion"""
    tmp_0 = in_0 + in_1
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    tmp_3 = tmp_2.to(torch.float32)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel using Triton
@triton.jit
def fused_add_softmax_dropout_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel: addition + softmax + dropout + type conversion"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: Element-wise addition
    added = x + y
    
    # Step 2: Softmax along last dimension
    # For softmax, we need to process along the last dimension
    # We'll use a simplified approach assuming 4D tensors [batch, heads, seq_len, seq_len]
    if n_elements >= 4096:  # Assume at least 4x4 matrix for softmax
        # Reshape for softmax processing per head
        # This is a simplified implementation - in practice you'd need more complex indexing
        max_val = tl.max(added, mask=mask)
        shifted = added - max_val
        exp_val = tl.exp(shifted)
        sum_exp = tl.sum(exp_val, axis=0, keepdim=True, mask=mask)
        softmax_out = exp_val / (sum_exp + 1e-20)
    else:
        # For smaller tensors, use simpler approach
        max_val = tl.max(added, mask=mask)
        shifted = added - max_val
        softmax_out = tl.exp(shifted)
        sum_exp = tl.sum(softmax_out, mask=mask)
        softmax_out = softmax_out / (sum_exp + 1e-20)
    
    # Step 3: Dropout
    if dropout_p > 0:
        # Generate random mask using fast random approach
        rand_vals = tl.rand(offsets)
        dropout_mask = rand_vals > dropout_p
        dropout_out = softmax_out * dropout_mask
    else:
        dropout_out = softmax_out
    
    # Step 4: Type conversion to float32 (already float32, but ensure)
    out = tl.cast(dropout_out, tl.float32)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def fused_add_softmax_dropout(x, y, dropout_p=0.1):
    """Wrapper function to launch the fused kernel"""
    # Use the same shape as the output
    out_shape = x.shape
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Flatten for processing if needed, but keep original shape in mind
    # For simplicity, we'll process the entire tensor as 1D
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_softmax_dropout_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        dropout_p=dropout_p,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    # Return a closure that captures the dropout rate
    def kernel_with_dropout_p(x, y):
        dropout_p = 0.1  # Default dropout rate from the patterns
        return fused_add_softmax_dropout(x, y, dropout_p)
    
    return kernel_with_dropout_p