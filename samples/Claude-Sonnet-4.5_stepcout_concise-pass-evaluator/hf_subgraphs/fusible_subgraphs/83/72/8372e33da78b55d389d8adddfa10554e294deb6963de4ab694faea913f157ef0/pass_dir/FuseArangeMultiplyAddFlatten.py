import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern to match: multiply -> view -> unsqueeze -> add -> view
    We skip matching arange since it has dynamic shape
    """
    # Match from the multiply operation onwards
    # in_2 here represents the result of torch.arange
    tmp_3 = in_2 * in_1
    tmp_4 = tmp_3.view((1,))
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 + in_0
    tmp_7 = tmp_6.view(-1)
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_kernel(
    in_0_ptr,
    out_ptr,
    offset_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes the entire pattern in one pass.
    For in_2=1, offset_val will be 0, so this is essentially a copy.
    But we avoid multiple kernel launches and intermediate allocations.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    in_0_val = tl.load(in_0_ptr + offsets, mask=mask, other=0)
    
    # Add the computed offset (which is 0 when in_2=1)
    result = in_0_val + offset_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_impl(in_0, in_1, in_2):
    """
    Optimized implementation: when offset is 0, just flatten directly
    in_2 is the result of torch.arange, in_1 is the multiplier scalar, in_0 is the indices
    """
    # Extract scalar values
    if isinstance(in_1, torch.Tensor):
        in_1_val = in_1.item()
    else:
        in_1_val = in_1
    
    # Compute offset from arange result
    if in_2.numel() == 1:
        offset_val = (in_2.item() * in_1_val)
    else:
        offset_val = 0
    
    # Fast path for offset == 0: just flatten
    if offset_val == 0:
        return in_0.view(-1)
    
    # Slow path: use Triton kernel for non-zero offset
    n_elements = in_0.numel()
    out = torch.empty(n_elements, dtype=in_0.dtype, device=in_0.device)
    
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_kernel[grid](
        in_0,
        out,
        offset_val,
        n_elements,
    )
    
    return out

def replacement_func():
    return fused_impl