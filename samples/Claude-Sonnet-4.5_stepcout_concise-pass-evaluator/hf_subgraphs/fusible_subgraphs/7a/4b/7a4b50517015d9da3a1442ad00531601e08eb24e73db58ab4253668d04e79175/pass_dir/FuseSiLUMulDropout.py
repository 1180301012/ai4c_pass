import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match: silu + multiply + dropout(p=0.0)
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_silu_mul_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: out = silu(in0) * in1
    where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute silu(in0) = in0 * sigmoid(in0)
    sigmoid_in0 = tl.sigmoid(in0)
    silu_in0 = in0 * sigmoid_in0
    
    # Multiply by in1
    out = silu_in0 * in1
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_silu_mul(in_0, in_1):
    """
    Wrapper function for the fused kernel
    """
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_silu_mul_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out


def replacement_func():
    return fused_silu_mul