import torch
import triton
import triton.language as tl


@triton.jit
def hardtanh_mul_kernel(
    in_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    HARDTANH_MIN: tl.constexpr,
    HARDTANH_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies hardtanh(in, min, max) and multiplies with other.
    hardtanh(x, min, max) = clamp(x, min, max) = max(min, min(max, x))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_val = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    other_val = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Apply hardtanh: clamp to [HARDTANH_MIN, HARDTANH_MAX]
    in_clamped = tl.minimum(tl.maximum(in_val, HARDTANH_MIN), HARDTANH_MAX)
    
    # Multiply
    out = in_clamped * other_val
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def hardtanh_mul_wrapper(in_tensor, other_tensor):
    """
    Wrapper for the fused hardtanh + multiply kernel.
    Computes: hardtanh(in_tensor, 0.0, 6.0) * other_tensor
    """
    n_elements = in_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(other_tensor)
    
    hardtanh_mul_kernel[(num_programs,)](
        in_ptr=in_tensor,
        other_ptr=other_tensor,
        out_ptr=out,
        n_elements=n_elements,
        HARDTANH_MIN=0.0,
        HARDTANH_MAX=6.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_2, in_1, in_0, in_3):
    """
    Match the pattern: conv2d(bias, weight, input) followed by hardtanh(input2) * conv2d
    """
    # Conv2d: torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Hardtanh on in_3
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    
    # Element-wise multiplication
    tmp_4 = tmp_3 * conv2d
    
    return tmp_4


def replacement_args(in_2, in_1, in_0, in_3):
    """
    Extract arguments needed for replacement.
    For the fused kernel, we only need:
    - in_3 (hardtanh input) -> in_tensor
    - conv2d result -> other_tensor
    """
    # We need to compute conv2d first to get the second tensor for multiplication
    conv2d_result = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return (in_3, conv2d_result)


def replacement_func():
    return hardtanh_mul_wrapper