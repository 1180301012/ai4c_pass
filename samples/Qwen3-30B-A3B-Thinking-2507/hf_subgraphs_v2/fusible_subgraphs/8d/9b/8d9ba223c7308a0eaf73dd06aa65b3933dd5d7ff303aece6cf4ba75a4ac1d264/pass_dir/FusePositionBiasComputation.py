import torch
import triton
import triton.language as tl

def pattern(T):
    p = T.permute(2, 0, 1)
    c = p.contiguous()
    s = torch.sigmoid(c)
    m = 16 * s
    return m

def replacement_args(T):
    return (T,)

@triton.jit
def bias_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.int32,
    D: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    m = offsets // (64 * 64)
    n = (offsets // 64) % 64
    k = offsets % 64
    input_idx = offsets
    x = tl.load(x_ptr + input_idx, mask=mask, other=0.0)
    s_val = tl.sigmoid(x.to(tl.float32)).to(tl.bfloat16)
    out_val = 16 * s_val
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def kernel_wrapper(T):
    D = T.shape[2]
    out = torch.empty([D, 64, 64], dtype=T.dtype, device=T.device)
    n_elements = D * 64 * 64
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    bias_kernel[(num_blocks,)](T, out, n_elements, D, BLOCK_SIZE)
    return out

def replacement_func():
    return kernel_wrapper