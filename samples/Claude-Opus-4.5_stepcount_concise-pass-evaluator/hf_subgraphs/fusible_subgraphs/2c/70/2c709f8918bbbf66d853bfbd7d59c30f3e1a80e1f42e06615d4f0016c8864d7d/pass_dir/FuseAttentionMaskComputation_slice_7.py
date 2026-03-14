import torch
import triton
import triton.language as tl

# Graph 2: chinese-clip-vit-huge-patch14 - slice(None, 7, None) = [:7]
def pattern(in_0, in_1):
    tmp_1 = in_1.to(dtype=torch.float32)
    tmp_2 = 1.0 - tmp_1
    tmp_3 = tmp_2 * -3.4028234663852886e+38
    tmp_4 = in_0[:, :7]
    return tmp_3, tmp_4

# Extract arguments for the replacement
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel for attention mask computation
@triton.jit
def fused_attention_mask_kernel(
    in_1_ptr,
    out_ptr,
    neg_inf: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask, other=0)
    in_1_float = in_1_val.to(tl.float32)
    out_val = (1.0 - in_1_float) * neg_inf
    
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_attention_mask_wrapper(in_0, in_1):
    in_1_shape = in_1.shape
    n_elements = in_1.numel()
    neg_inf = -3.4028234663852886e+38
    
    out = torch.empty_like(in_1, dtype=torch.float32)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_mask_kernel[(num_programs,)](
        in_1_ptr=in_1,
        out_ptr=out,
        neg_inf=neg_inf,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    out = out.reshape(in_1_shape)
    sliced_in_0 = in_0[:, :7]
    
    return out, sliced_in_0

def replacement_func():
    return fused_attention_mask_wrapper