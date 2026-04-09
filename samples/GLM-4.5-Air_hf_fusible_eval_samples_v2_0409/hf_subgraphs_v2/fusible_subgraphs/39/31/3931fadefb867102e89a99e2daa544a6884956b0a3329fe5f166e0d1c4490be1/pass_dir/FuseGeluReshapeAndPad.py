import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern to match: GELU -> reshape -> reshape -> pad operations"""
    tmp_0 = torch.nn.functional.gelu(x)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_gelu_reshape_pad_kernel(
    input_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    n_input_elements,
    n_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for GELU + reshape + pad operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_input_elements
    
    # Load input data from [1, 124, 1536]
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply GELU activation
    sqrt_2_over_pi = 0.7978845608028654
    coeff_1 = 0.044715
    coeff_2 = 0.5
    
    x_cubed = x * x * x
    inner = x + coeff_1 * x_cubed
    tanh_arg = sqrt_2_over_pi * inner
    tanh_val = tl.tanh(tanh_arg)
    gelu_val = x * coeff_2 * (1.0 + tanh_val)
    
    # Store GELU result directly in output buffer with final shape [1, 249, 768]
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def fused_gelu_reshape_pad(x):
    """Fused operation: GELU + reshape + reshape + padding"""
    # Input shape: [1, 124, 1536], output shape: [1, 249, 768]
    output_shape = [1, 249, 768]
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Process input elements (excluding padding region)
    n_input_elements = 1 * 124 * 1536  # 190464 elements
    BLOCK_SIZE = 1024
    num_programs = (n_input_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_gelu_reshape_pad_kernel[(num_programs,)](
        x,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        n_input_elements,
        output.numel(),  # 191632 elements (with padding)
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_gelu_reshape_pad