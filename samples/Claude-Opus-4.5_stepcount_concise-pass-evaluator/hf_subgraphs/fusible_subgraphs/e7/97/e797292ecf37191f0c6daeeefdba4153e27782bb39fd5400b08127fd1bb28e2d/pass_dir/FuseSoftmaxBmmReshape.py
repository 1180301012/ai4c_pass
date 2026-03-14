import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match just bmm
    in_0 is the output of dropout (softmax output)
    """
    result = torch.bmm(in_0, in_1)
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_bmm_kernel(
    attn_ptr,       # [num_heads, 1, 1] - already softmaxed
    value_ptr,      # [num_heads, 1, head_dim]
    out_ptr,        # [num_heads, 1, head_dim]
    num_heads,
    head_dim,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for bmm
    """
    pid = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute which head each element belongs to
    head_idx = offsets // head_dim
    local_idx = offsets % head_dim
    
    # Load attention values (already softmaxed)
    attn_vals = tl.load(attn_ptr + head_idx, mask=mask, other=0.0)
    
    # Load value elements
    value_vals = tl.load(value_ptr + head_idx * head_dim + local_idx, mask=mask, other=0.0)
    
    # Compute output: bmm result
    out_vals = value_vals * attn_vals
    
    # Store to output
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@torch.fx.wrap
def fused_bmm(in_0, in_1):
    """
    Fused implementation of bmm
    
    in_0: [num_heads, 1, 1] - already softmaxed attention weights
    in_1: [num_heads, 1, head_dim] - value states
    output: tensor of shape [num_heads, 1, head_dim]
    """
    num_heads = in_0.shape[0]
    head_dim = in_1.shape[2]
    total_elements = num_heads * head_dim
    
    # Ensure inputs are contiguous
    in_0_contig = in_0.contiguous()
    in_1_contig = in_1.contiguous()
    
    # Allocate output - same shape as bmm output
    out = torch.empty((num_heads, 1, head_dim), dtype=in_1.dtype, device=in_1.device)
    
    # Compute grid
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    fused_bmm_kernel[grid](
        in_0_contig,
        in_1_contig,
        out,
        num_heads,
        head_dim,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_bmm