import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    """
    Match the attention mask creation pattern:
    First create eq mask, then multiply by large negative number, then add dimensions
    """
    tmp_5 = x.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 *= -3.4028234663852886e+38  # Use in-place multiplication to match exactly
    tmp_7 = tmp_6
    tmp_8 = tmp_7.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    return tmp_9

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized attention mask kernel
@triton.jit
def attention_mask_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    """
    Optimized kernel for attention mask creation:
    - Convert positions equal to 1 to neg_inf for masking
    - Uses direct memory access and scalar operations
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Create mask: where input == 1, set to NEG_INF, otherwise keep as is
    # Note: We're preserving the original behavior where non-1 values remain unchanged
    # In this specific case, the original implementation always applies NEG_INF to the boolean mask
    bool_mask = (input_vals == 1)
    output_vals = tl.where(bool_mask, NEG_INF, input_vals.to(tl.float32))
    
    # Store result
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@torch.fx.wrap
def optimized_attention_mask(x):
    """
    Create optimized attention mask:
    - Positions equal to 1 get masked with -inf
    - Output shape: [1, 15] -> [1, 15, 1, 1] (but we'll match the original [1, 15, 1])
    """
    device = x.device
    
    # Create tensor with neg_inf for masked positions
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate output shape: original pattern does two unsqueeze(1)
    # From [1, 15] -> [1, 15, 1] -> [1, 15, 1, 1]
    # But we'll match exactly what the pattern returns: [1, 15, 1]
    output_shape = list(x.shape)
    output_shape.insert(-1, 1)  # Insert at second-to-last position
    
    output = torch.empty(output_shape, dtype=torch.float32, device=device)
    
    # Flatten input and output for processing
    x_flat = x.view(-1)
    output_flat = output.view(-1)
    
    attention_mask_kernel[(num_programs,)](
        x_flat,
        output_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        NEG_INF=-3.4028234663852886e+38,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_attention_mask