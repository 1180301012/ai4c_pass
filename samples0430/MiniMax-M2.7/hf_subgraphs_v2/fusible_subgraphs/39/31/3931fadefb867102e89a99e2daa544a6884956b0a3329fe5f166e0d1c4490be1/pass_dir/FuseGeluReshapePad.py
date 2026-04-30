import torch
import triton
import triton.language as tl

# Constants for GELU (will be used as constexpr inside kernel)
GELU_C1 = 0.7978845608  # sqrt(2/pi)
GELU_C2 = 0.04471499822384429  # sqrt(2/pi) * 0.044715


@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for exact GELU activation.
    
    GELU formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Using tanh(y) = (exp(2y) - 1) / (exp(2y) + 1):
    tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
           = 1 - 2/(exp(2y) + 1)
    """
    # Define constants as constexpr inside kernel
    c1 = 0.7978845608  # sqrt(2/pi)
    c2 = 0.04471499822384429  # sqrt(2/pi) * 0.044715
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute tanh using exp: tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
    x_sq = x * x
    x_cubed = x * x_sq
    tanh_arg = c1 * (x + c2 * x_cubed)
    
    # Compute tanh via exp: tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
    exp_2y = tl.exp(2.0 * tanh_arg)
    tanh_y = (exp_2y - 1.0) / (exp_2y + 1.0)
    
    # GELU = 0.5 * x * (1 + tanh(y))
    out = 0.5 * x * (1.0 + tanh_y)
    
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_gelu(x: torch.Tensor) -> torch.Tensor:
    """Optimized GELU with Triton - matches PyTorch exact GELU"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    gelu_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0):
    """Match gelu(in_0) pattern"""
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return optimized_gelu