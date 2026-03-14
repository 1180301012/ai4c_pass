import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Pattern matching for RoPE cos/sin computation:
    cat -> cos -> mul -> to(bfloat16)
    cat -> sin -> mul -> to(bfloat16)
    """
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return (tmp_6, tmp_7)


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def rope_cos_sin_kernel(
    in_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for RoPE cos/sin computation.
    Reads input once, computes cos and sin, writes both outputs.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input (will be duplicated via cat operation)
    # Each element appears twice in the concatenated result
    half_n = n_elements // 2
    input_offsets = offsets % half_n
    x = tl.load(in_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute cos and sin
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store results (already in bfloat16 via pointer dtype)
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)


@torch.fx.wrap
def fused_rope_cos_sin(in_1):
    """
    Fused RoPE cos/sin computation.
    Input: [B, S, D] float32
    Output: cos [B, S, 2*D] bfloat16, sin [B, S, 2*D] bfloat16
    """
    # Output shape after cat is double the last dimension
    out_shape = list(in_1.shape)
    out_shape[-1] *= 2
    
    cos_out = torch.empty(out_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_out = torch.empty(out_shape, dtype=torch.bfloat16, device=in_1.device)
    
    n_elements = cos_out.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    rope_cos_sin_kernel[grid](
        in_1,
        cos_out,
        sin_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, sin_out


def replacement_func():
    return fused_rope_cos_sin