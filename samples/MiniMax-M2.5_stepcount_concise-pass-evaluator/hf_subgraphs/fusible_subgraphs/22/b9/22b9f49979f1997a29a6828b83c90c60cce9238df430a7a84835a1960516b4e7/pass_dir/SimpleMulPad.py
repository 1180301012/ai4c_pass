import torch
import triton
import triton.language as tl


def pattern(a, b):
    """
    Match: multiply followed by pad
    """
    mul_result = a * b
    pad_result = torch.nn.functional.pad(mul_result, (0, 0, 1, 0, 0, 0), 'constant', None)
    return (pad_result,)


def replacement_args(a, b):
    return (a, b)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def mul_pad_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    out = a * b
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def mul_pad_kernel_wrapper(a, b):
    # Handle the padding by creating output with extra row
    # Input: [..., N, K] -> Output: [..., N+1, K] with first row being zeros
    a_shape = a.shape
    
    # Create output with padded first dimension (add 1 to second to last dim)
    output_shape = list(a_shape)
    output_shape[-2] = output_shape[-2] + 1  # Add 1 row at top
    output = torch.zeros(output_shape, dtype=a.dtype, device=a.device)
    
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Multiply in Triton
    out_flat = torch.empty_like(a)
    mul_pad_kernel[(num_programs,)](
        a_ptr=a.flatten(),
        b_ptr=b.flatten(),
        out_ptr=out_flat.flatten(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Manual padding: copy to rows 1 onwards (row 0 stays zeros)
    # Reshape flat output back to original shape, then copy to padded output
    out_reshaped = out_flat.view(a_shape)
    output[..., 1:, :] = out_reshaped
    
    return output


def replacement_func():
    return mul_pad_kernel_wrapper