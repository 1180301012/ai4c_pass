import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, y):
    gelu_result = torch.nn.functional.gelu(x, approximate='none')
    fused = gelu_result * y
    return fused

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Triton kernel for fused gelu and multiplication
@triton.jit
def fused_gelu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = tl.cast(x, tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask)
    y_fp32 = tl.cast(y, tl.float32)

    # Compute gelu(x) * y using standard approximation (in float32)
    x = x_fp32
    y = y_fp32
    # Using common tanh-based approximation for gelu
    # Formula: x * 0.5 * (1 + tanh(0.79788456 * (x + 0.044715 * x**3)))
    x3 = x * x * x
    x3 = x3 * 0.044715
    x = x + x3
    x = x * 0.79788456
    exp_x = tl.exp(x)
    exp_neg_x = tl.exp(-x)
    tanh_x = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    gelu_x = x * 0.5 * (1 + tanh_x)
    out = tl.cast(gelu_x * y, x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_gelu_mul(x, y):
    n = x.numel()
    block_size = 1024
    num_blocks = (n + block_size - 1) // block_size
    out = torch.empty_like(x)
    fused_gelu_mul_kernel[(num_blocks,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=block_size,
    )
    return out

# Replacement function
def replacement_func():
    return fused_gelu_mul