import torch
import triton
import triton.language as tl

# Pattern matching function for addition + softmax + zero dropout + type conversion
def pattern(in_0, in_1):
    """Match the sequence: addition -> softmax -> dropout(0.0) -> type conversion"""
    tmp_0 = in_0 + in_1
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)  # dropout_p=0.0
    tmp_3 = tmp_2.to(torch.float32)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel for zero dropout case (no dropout operation)
@triton.jit
def fused_add_softmax_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel: addition + softmax (dropout is no-op when p=0.0)"""
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
    # Since dropout_p=0.0, we skip the dropout operation entirely
    if n_elements >= 4096:
        # For larger tensors, use optimized softmax
        max_val = tl.max(added, mask=mask)
        shifted = added - max_val
        exp_val = tl.exp(shifted)
        sum_exp = tl.sum(exp_val, axis=0, keepdim=True, mask=mask)
        softmax_out = exp_val / (sum_exp + 1e-20)
    else:
        # For smaller tensors
        max_val = tl.max(added, mask=mask)
        shifted = added - max_val
        softmax_out = tl.exp(shifted)
        sum_exp = tl.sum(softmax_out, mask=mask)
        softmax_out = softmax_out / (sum_exp + 1e-20)
    
    # Step 3: Type conversion to float32 (already float32, but ensure)
    out = tl.cast(softmax_out, tl.float32)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_softmax(x, y):
    """Wrapper function to launch the fused kernel (no dropout)"""
    out_shape = x.shape
    out = torch.empty(out_shape, dtype=torch.float32, device=x.device)
    
    # Process entire tensor as 1D for simplicity
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel (no dropout parameter)
    fused_add_softmax_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    # Return a closure that performs addition + softmax without dropout
    def kernel_no_dropout(x, y):
        return fused_add_softmax(x, y)
    
    return kernel_no_dropout