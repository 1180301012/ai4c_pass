import torch
import triton
import triton.language as tl


@triton.jit
def triton_rope_kernel(
    x_ptr, cos_ptr, sin_ptr, output_ptr,
    n_elements, n_half,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized RoPE kernel using 1D grid for simplicity."""
    pid = tl.program_id(0)
    half_size = n_half
    
    # Each thread handles one element in the first half and one in the second half
    # Process elements in first half
    offsets_first = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_second = offsets_first + half_size
    
    mask_first = offsets_first < half_size
    mask_second = offsets_second < n_elements
    
    # Load values for first half position
    x1 = tl.load(x_ptr + offsets_first, mask=mask_first, other=0.0)
    cos1 = tl.load(cos_ptr + offsets_first, mask=mask_first, other=0.0)
    sin1 = tl.load(sin_ptr + offsets_first, mask=mask_first, other=0.0)
    # Load rotated value
    x2_rot = tl.load(x_ptr + offsets_second, mask=mask_first, other=0.0)
    
    # Load values for second half position  
    x2 = tl.load(x_ptr + offsets_second, mask=mask_second, other=0.0)
    cos2 = tl.load(cos_ptr + offsets_second, mask=mask_second, other=0.0)
    sin2 = tl.load(sin_ptr + offsets_second, mask=mask_second, other=0.0)
    # Load rotated value for second half
    x1_rot = tl.load(x_ptr + offsets_first, mask=mask_second, other=0.0)
    
    # RoPE computation
    # First half: x * cos - x_rotated * sin
    out_first = x1 * cos1 - x2_rot * sin1
    # Second half: x * cos + x_rotated * sin  
    out_second = x2 * cos2 + x1_rot * sin2
    
    # Store
    tl.store(output_ptr + offsets_first, out_first, mask=mask_first)
    tl.store(output_ptr + offsets_second, out_second, mask=mask_second)


@torch.fx.wrap
def fused_rope_reshape(key_states, cos, sin, num_heads=8):
    """Fused RoPE + reshape for key states."""
    batch, head_in, seq, dim = key_states.shape
    n_half = dim // 2
    n_elements = batch * head_in * seq * dim
    
    # Run RoPE kernel
    rope_out = torch.empty_like(key_states)
    BLOCK_SIZE = 64
    num_programs = (n_half + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    triton_rope_kernel[(num_programs,)](
        key_states, cos, sin, rope_out,
        n_elements, n_half,
        key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape via unsqueeze + expand + reshape
    reshaped = rope_out.unsqueeze(1).expand(batch, num_heads, seq, dim).reshape(batch, num_heads, seq, dim)
    
    return rope_out, reshaped.contiguous()


def pattern(in_2, in_1, in_4):
    """
    Match the RoPE computation + reshape pattern.
    Operations matched:
    - tmp_0 = in_2 * in_1
    - tmp_1 = in_2[..., :128]
    - tmp_2 = in_2[..., 128:]
    - tmp_3 = -tmp_2
    - tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    - tmp_5 = tmp_4 * in_4
    - tmp_6 = tmp_0 + tmp_5
    - tmp_7 = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    - tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    - tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    
    Returns tmp_6 and tmp_9 (both outputs).
    """
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    tmp_7 = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    return tmp_6, tmp_9


def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)


def replacement_func():
    return fused_rope_reshape