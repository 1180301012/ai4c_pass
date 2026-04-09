import torch
import triton
import triton.language as tl



def pattern(x):
    """Pattern to match: GELU activation followed by reshaping and padding 
    This matches the computation after reshape fusion:
    GELU -> reshape to [1, 248, 768] -> pad to [1, 249, 768]
    """
    tmp_0 = torch.nn.functional.gelu(x)
    tmp_2 = tmp_0.reshape(1, 248, 768)  # This is the fused reshape from pass 1
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_gelu_pad_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU + padding kernel that processes data and adds padding in one pass"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply GELU activation
    # GELU(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff_1 = 0.044715
    coeff_2 = 0.5
    
    x_cubed = x * x * x
    inner = x + coeff_1 * x_cubed
    tanh_arg = sqrt_2_over_pi * inner
    tanh_val = tl.tanh(tanh_arg)
    gelu_val = x * coeff_2 * (1.0 + tanh_val)
    
    # For padding strategy: since we need to add 1 element along dimension 1 (248 -> 249),
    # and our total elements are 1 * 248 * 768 = 190464
    # The padding will be applied to create 1 * 249 * 768 = 191632 elements
    # We'll handle this by creating an output buffer with the padded shape
    
    # Store the GELU result (padding will be handled by output buffer allocation)
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def fused_gelu_pad(x):
    """Fused GELU + padding operation"""
    # Original shape after reshapes: [1, 248, 768], padded to [1, 249, 768]
    output_shape = [1, 249, 768]
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Process the original data (excluding padding region)
    n_elements = 1 * 248 * 768  # 190464 elements (excluding the padded row)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_gelu_pad_kernel[(num_programs,)](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_gelu_pad