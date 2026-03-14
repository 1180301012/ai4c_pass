import torch
import triton
import triton.language as tl

def pattern(attention_mask):
    """Pattern matches attention mask computation: (1.0 - attention_mask.to(float32)) * -FLT_MAX"""
    tmp_1 = attention_mask.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    return tmp_3

def replacement_args(attention_mask):
    """Extract arguments needed for the fusion kernel"""
    return (attention_mask,)

@triton.jit
def attention_mask_fusion_kernel(
    attention_mask_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for attention mask computation: (1.0 - x) * -FLT_MAX"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load attention mask
    attention_mask = tl.load(attention_mask_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: (1.0 - attention_mask) * -FLT_MAX
    # This is a common pattern for attention masking where we create large negative values
    # for padding tokens to effectively zero them out in softmax
    neg_inf = -3.4028234663852886e+38
    result = (1.0 - attention_mask) * neg_inf
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_attention_mask(attention_mask):
    """Wrapper function to launch the fused attention mask kernel"""
    # Get input shape and compute total elements
    if attention_mask.dim() == 4:  # Shape like [batch, 1, 1, seq_len]
        # For 4D tensors, we need to process all elements but the singleton dimensions can be flattened
        n_elements = attention_mask.numel()
        output_shape = attention_mask.shape
    else:  # Other tensor shapes
        n_elements = attention_mask.numel()
        output_shape = attention_mask.shape
    
    # Create output tensor with same shape and float32 dtype
    output = torch.empty_like(attention_mask, dtype=torch.float32)
    
    # Block size optimized for attention mask computation
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    attention_mask_fusion_kernel[(num_programs,)](
        attention_mask_ptr=attention_mask,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused attention mask function"""
    return fused_attention_mask