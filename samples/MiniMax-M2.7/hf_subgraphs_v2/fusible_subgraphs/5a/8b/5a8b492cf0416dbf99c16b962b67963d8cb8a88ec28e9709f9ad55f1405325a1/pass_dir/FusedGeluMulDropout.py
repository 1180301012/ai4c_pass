import torch
import triton
import triton.language as tl


# Autotune configurations - optimized for various tensor sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_mul_dropout_kernel(
    x_ptr,       # in_0 (gated input)
    y_ptr,       # in_1 (non-gated input)
    out_ptr,     # output tensor
    n_elements,  # total number of elements
    seed,        # random seed for dropout
    DROPOUT_P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: gelu(x) * y with dropout
    GELU formula (approximate='none'): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    # Get current program/thread block
    pid = tl.program_id(0)
    
    # Compute offsets for this block
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load x (gated input)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Constants for GELU
    GELU_C1 = 0.7978845608028654  # sqrt(2/pi)
    GELU_C2 = 0.044715
    
    x_cubed = x * x * x
    inner = GELU_C1 * (x + GELU_C2 * x_cubed)
    
    # Compute tanh using sigmoid formula: tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    two_z = 2.0 * inner
    exp_2z = tl.exp(two_z)
    tanh_inner = (exp_2z - 1.0) / (exp_2z + 1.0)
    
    gelu_out = 0.5 * x * (1.0 + tanh_inner)
    
    # Load y (non-gated input)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    
    # Multiply: gelu_out * y
    mul_out = gelu_out * y
    
    # Generate dropout mask
    rng_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    random_float = tl.rand(seed, rng_offset)
    
    # Dropout: keep probability is (1 - DROPOUT_P)
    keep_mask = random_float > DROPOUT_P
    
    # Scale factor = 1 / (1 - DROPOUT_P) = 1 / 0.9 = 1.111...
    SCALE = 1.1111111111111112  # 1 / (1 - 0.1)
    
    # Apply dropout
    dropout_out = tl.where(keep_mask, mul_out * SCALE, 0.0)
    
    # Store result
    tl.store(out_ptr + offs, dropout_out, mask=mask)


@torch.fx.wrap
def fused_gelu_mul_dropout(x, y, dropout_p=0.1, training=True, seed=42):
    """
    Fused GELU + Multiply + Dropout operation
    
    Args:
        x: gated input tensor (shape: [..., hidden_dim])
        y: non-gated input tensor (shape: [..., hidden_dim])
        dropout_p: dropout probability (default 0.1)
        training: whether in training mode (default True)
        seed: random seed for dropout
    
    Returns:
        Tensor after gelu(x) * y with dropout applied
    """
    n_elements = x.numel()
    
    # Autotuning will pick the best BLOCK_SIZE and NUM_ELTS
    # Use a default that's reasonable for most cases
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure minimum number of programs for parallelism
    if num_programs < 4:
        num_programs = 4
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Launch the fused kernel
    gelu_mul_dropout_kernel[(num_programs,)](
        x,
        y,
        output,
        n_elements,
        seed,
        dropout_p,
    )
    
    return output


def pattern(in_0, in_1):
    """
    Match the pattern: gelu(in_0) * in_1 with dropout
    """
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement function.
    """
    return (in_0, in_1, 0.1, True, 42)


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_gelu_mul_dropout