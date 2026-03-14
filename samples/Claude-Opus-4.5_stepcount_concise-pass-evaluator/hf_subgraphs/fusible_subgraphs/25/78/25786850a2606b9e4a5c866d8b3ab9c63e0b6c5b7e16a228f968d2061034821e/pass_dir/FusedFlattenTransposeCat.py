import torch
import triton
import triton.language as tl

# Fuse flatten(2) + transpose(1,2) + expand + cat + dropout(0.0)
# This combines all post-conv2d operations into a single optimized kernel

def pattern(conv_out, cls_token):
    tmp_6 = conv_out.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = cls_token.expand(1, -1, -1)
    tmp_9 = torch.cat((tmp_8, tmp_7), dim=1)
    tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False)
    return tmp_10


def replacement_args(conv_out, cls_token):
    return (conv_out, cls_token)


@triton.jit
def fused_kernel_optimized(
    conv_ptr,
    cls_ptr, 
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel with better memory access patterns
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Constants  
    C: tl.constexpr = 768
    S: tl.constexpr = 196
    total: tl.constexpr = 151296
    
    mask = offsets < total
    
    # Compute indices
    is_cls = offsets < C
    
    # For cls region
    cls_idx = offsets
    
    # For conv region - transpose [768, 196] -> [196, 768]
    adj = offsets - C
    s = adj // C  # spatial index
    c = adj % C   # channel index
    conv_idx = c * S + s
    
    # Vectorized load with proper masking
    cls_vals = tl.load(cls_ptr + cls_idx, mask=mask & is_cls, other=0.0)
    conv_vals = tl.load(conv_ptr + conv_idx, mask=mask & ~is_cls, other=0.0)
    
    # Select and store
    vals = tl.where(is_cls, cls_vals, conv_vals)
    tl.store(out_ptr + offsets, vals, mask=mask)


@torch.fx.wrap
def fused_flatten_transpose_cat_dropout(conv_out, cls_token):
    # Pre-allocate output
    out = torch.empty(1, 197, 768, dtype=conv_out.dtype, device=conv_out.device)
    
    # Use larger block size for better throughput
    BLOCK_SIZE = 2048
    n_elements = 151296
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_kernel_optimized[(num_blocks,)](
        conv_out.view(-1),
        cls_token.view(-1),
        out.view(-1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_flatten_transpose_cat_dropout