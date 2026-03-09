import torch
import triton
import triton.language as tl


# Pattern matching function - matches gelu followed by flatten(1, -1)
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


# Extract arguments for replacement
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def gelu_flatten_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused gelu + flatten kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # For flatten(1, -1), output is [B, C*H*W]
    # Load from original [B, C, H, W] tensor
    c_h_w = C * H * W
    b = offsets // c_h_w
    remainder = offsets % c_h_w
    c = remainder // (H * W)
    remainder = remainder % (H * W)
    h = remainder // W
    w = remainder % W
    
    # Calculate flat index in original tensor
    flat_idx = b * c_h_w + c * (H * W) + h * W + w
    
    # Load
    x = tl.load(input_ptr + flat_idx, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
    # Using: tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    sqrt_2_over_pi = 0.7978845608028654
    alpha = 0.044715
    x_sq = x * x
    arg = sqrt_2_over_pi * x * (1.0 + alpha * x_sq)
    exp_2arg = tl.exp(2.0 * arg)
    tanh_val = (exp_2arg - 1.0) / (exp_2arg + 1.0)
    gelu_out = x * 0.5 * (1.0 + tanh_val)
    
    # Store
    tl.store(output_ptr + offsets, gelu_out, mask=mask)


@torch.fx.wrap
def gelu_flatten_wrapper(in_0):
    """Wrapper function for fused gelu + flatten."""
    B, C, H, W = in_0.shape
    num_elements = B * C * H * W
    output_shape = (B, C * H * W)
    
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    gelu_flatten_kernel[(num_programs,)](
        in_0,
        out,
        num_elements,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return gelu_flatten_wrapper