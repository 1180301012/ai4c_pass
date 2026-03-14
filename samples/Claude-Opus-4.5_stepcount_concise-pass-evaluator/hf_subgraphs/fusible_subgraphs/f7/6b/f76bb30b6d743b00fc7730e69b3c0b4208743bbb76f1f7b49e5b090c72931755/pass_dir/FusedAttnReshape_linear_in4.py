import torch
import triton
import triton.language as tl

# Match query/value path: view -> transpose -> reshape
# This transforms [1, seq, 1024] -> [16, seq, 64]
def pattern(x):
    t1 = x.view(1, -1, 16, 64)
    t2 = t1.transpose(1, 2)
    out = t2.reshape(16, -1, 64)
    return out


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_view_transpose_reshape_kernel(
    in_ptr, out_ptr,
    seq_len,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused view + transpose + reshape operation.
    Input: [1, seq, 1024] contiguous where 1024 = 16 * 64
    Output: [16, seq, 64] contiguous
    
    Transformation: in[0, s, h*64+d] -> out[h, s, d]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Output is [16, seq, 64], compute indices
    h = offsets // (seq_len * 64)
    s = (offsets // 64) % seq_len
    d = offsets % 64
    
    # Input is [1, seq, 1024], compute linear index
    # in[0, s, h*64+d]
    in_idx = s * 1024 + h * 64 + d
    
    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def triton_fused_qv_reshape(x):
    # Input x has shape [1, seq, 1024] contiguous
    # Output should be [16, seq, 64] contiguous
    
    seq_len = x.shape[1]
    out = x.new_empty((16, seq_len, 64))
    
    n_elements = 16 * seq_len * 64
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_view_transpose_reshape_kernel[grid](
        x, out, seq_len, n_elements
    )
    
    return out


def replacement_func():
    return triton_fused_qv_reshape