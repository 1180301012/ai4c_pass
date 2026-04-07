import torch
import triton
import triton.language as tl

def pattern(attention_mask):
    """
    Pattern for attention mask processing that needs optimization.
    Matches: tmp_12 = attention_mask.to(dtype=torch.float32); 
             tmp_13 = 1.0 - tmp_12; 
             tmp_14 = tmp_13 * -3.4028234663852886e+38
    """
    tmp_12 = attention_mask.to(dtype=torch.float32)
    tmp_13 = 1.0 - tmp_12
    tmp_14 = tmp_13 * -3.4028234663852886e+38
    return tmp_14

def replacement_args(attention_mask):
    return (attention_mask,)

@triton.jit
def attention_mask_kernel(
    mask_ptr,
    output_ptr,
    n_elements,
    large_neg_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for optimized attention mask processing.
    Converts to float32, inverts, and multiplies by large negative value.
    """
    pid = tl.program_id(0)
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load attention mask values
    mask_values = tl.load(mask_ptr + offsets, mask=mask, other=1.0)
    
    # Convert to float32, invert (1.0 - x), multiply by large negative value
    # This can be done in one operation to reduce memory access
    result = (1.0 - mask_values.to(tl.float32)) * large_neg_val
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def attention_mask_wrapper(attention_mask):
    """
    Wrapper function to launch the optimized attention mask kernel.
    """
    n_elements = attention_mask.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Large negative value for padding tokens
    large_neg_val = -3.4028234663852886e+38
    
    output = torch.empty_like(attention_mask, dtype=torch.float32)
    
    attention_mask_kernel[(num_programs,)](
        mask_ptr=attention_mask,
        output_ptr=output,
        n_elements=n_elements,
        large_neg_val=large_neg_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """
    Returns the optimized attention mask function.
    """
    return attention_mask_wrapper