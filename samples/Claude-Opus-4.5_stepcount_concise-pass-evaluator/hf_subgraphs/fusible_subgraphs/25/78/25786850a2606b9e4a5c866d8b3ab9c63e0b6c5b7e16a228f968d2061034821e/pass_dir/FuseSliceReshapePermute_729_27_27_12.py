import torch
import triton
import triton.language as tl

# Pattern: flatten(2) -> transpose(1, 2)
# Input: [1, 768, 14, 14] -> [1, 768, 196] -> [1, 196, 768]

def pattern(x):
    tmp_6 = x.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7


def replacement_args(x):
    return (x,)


@triton.jit
def flatten_transpose_kernel(
    in_ptr,
    out_ptr,
    C: tl.constexpr,  # 768
    S: tl.constexpr,  # 196
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for flatten + transpose
    Input: [1, 768, 14, 14] = [1, 768, 196] after flatten
    Output: [1, 196, 768]
    
    Input layout (row-major): idx = c * 196 + s
    Output layout (row-major): idx = s * 768 + c
    
    For coalesced writes, we process output linearly:
    - out_idx -> s = out_idx // 768, c = out_idx % 768
    - in_idx = c * 196 + s
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Output index decomposition
    s = offsets // C  # spatial index (0-195)
    c = offsets % C   # channel index (0-767)
    
    # Input index
    in_idx = c * S + s
    
    # Load from input and store to output (coalesced writes)
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def fused_flatten_transpose(x):
    """
    Fused implementation of:
    - x.flatten(2)
    - .transpose(1, 2)
    
    Input: [1, 768, 14, 14]
    Output: [1, 196, 768]
    """
    B, C, H, W = x.shape
    S = H * W  # 196
    n_elements = B * S * C  # 150528
    
    out = torch.empty(B, S, C, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    flatten_transpose_kernel[(num_programs,)](
        in_ptr=x,
        out_ptr=out,
        C=C,
        S=S,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_flatten_transpose