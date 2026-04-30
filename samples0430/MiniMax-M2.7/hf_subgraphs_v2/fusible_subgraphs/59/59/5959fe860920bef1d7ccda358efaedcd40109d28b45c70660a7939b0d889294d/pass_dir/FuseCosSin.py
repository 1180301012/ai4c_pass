import torch
import triton
import triton.language as tl

@triton.jit
def cat_twice_kernel(src_ptr, dst_ptr, n_elements, stride, BLOCK_SIZE: tl.constexpr):
    """Copy tensor twice to create concatenated result"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    # Store to both halves
    tl.store(dst_ptr + offsets, x, mask=mask)
    tl.store(dst_ptr + offsets + stride, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_cos_sin_kernel(
    x_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cos and sin in one go (reduces memory bandwidth)
    cos_out = tl.cos(x)
    sin_out = tl.sin(x)
    
    # Store outputs
    tl.store(cos_out_ptr + offsets, cos_out, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_out, mask=mask)


def cossin_wrapper(in_0, in_1, in_2):
    """Module-level wrapper function for cos/sin pass"""
    # Create concatenated tensor manually using torch.empty
    in_1_shape = in_1.shape
    new_shape = list(in_1_shape)
    new_shape[-1] = new_shape[-1] * 2
    
    # Create output tensors (only tensor allocation is allowed)
    x_concat = torch.empty(new_shape, dtype=in_1.dtype, device=in_1.device)
    cos_out = torch.empty(new_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_out = torch.empty(new_shape, dtype=torch.bfloat16, device=in_1.device)
    
    n_elements = x_concat.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Copy in_1 to both halves using Triton kernel
    src_elements = in_1.numel()
    cat_twice_kernel[(num_programs,)](
        src_ptr=in_1,
        dst_ptr=x_concat,
        n_elements=src_elements,
        stride=src_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Now apply cos/sin kernel
    fused_cos_sin_kernel[(num_programs,)](
        x_ptr=x_concat,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, sin_out


def pattern(in_0, in_1, in_2):
    """
    Match the cat + cos/sin pattern:
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=target_dtype)
    tmp_7 = tmp_5.to(dtype=target_dtype)
    """
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    
    return tmp_6, tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return cossin_wrapper