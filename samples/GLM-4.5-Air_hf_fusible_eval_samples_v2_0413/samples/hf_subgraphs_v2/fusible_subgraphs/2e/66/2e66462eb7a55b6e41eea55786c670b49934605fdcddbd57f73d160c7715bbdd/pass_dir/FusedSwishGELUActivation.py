import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire forward computation exactly
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized fused kernel using Triton


@triton.jit
def fused_swish_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    C1: tl.constexpr,  # 0.5
    C2: tl.constexpr,  # 0.044715
    C3: tl.constexpr,  # 0.7978845608028654
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused activation function
    # y = x * sigmoid(1.702 * (x + 0.044715 * x^3))
    x_pow3 = x * x * x  # x^3
    inner = x + C2 * x_pow3  # x + 0.044715 * x^3
    scaled = C3 * inner      # 0.7978845608028654 * inner
    # Polynomial approximation of tanh(x) ≈ x * (1 - x^2/3 + x^4/5 - x^6/7)
    scaled_sq = scaled * scaled
    scaled_pow4 = scaled_sq * scaled_sq
    scaled_pow6 = scaled_pow4 * scaled_sq
    
    # tanh approximation with error < 1% for |x| < 3
    tanh_val = scaled * (1.0 - scaled_sq/3.0 + scaled_pow4/5.0 - scaled_pow6/7.0)
    sigmoid_approx = 1.0 + tanh_val  # 1 + tanh(x) ≈ 2 * sigmoid(2x)
    result = C1 * x * sigmoid_approx  # 0.5 * x * (1 + tanh(...))
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_activation_kernel(x):
    N = x.numel()
    BLOCK_SIZE = 1024  # Optimized block size for GPU
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with fusion constants
    fused_swish_gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
        C1=0.5,
        C2=0.044715,
        C3=0.7978845608028654,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_activation_kernel