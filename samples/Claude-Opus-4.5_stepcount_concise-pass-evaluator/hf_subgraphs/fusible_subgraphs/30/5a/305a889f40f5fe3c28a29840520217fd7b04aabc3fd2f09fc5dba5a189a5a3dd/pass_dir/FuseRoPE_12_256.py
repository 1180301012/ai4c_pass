import torch
import triton
import triton.language as tl

# Pattern for eva02_base_patch14_224 - reshape((1, 12, 256, 64))
def pattern(neg_input, slice_input, prefix, mul_base, sin_emb, ref_tensor):
    tmp_1 = -neg_input
    tmp_2 = slice_input[Ellipsis, slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    tmp_4 = tmp_3.reshape((1, 12, 256, 64))
    tmp_5 = tmp_4 * sin_emb
    tmp_6 = mul_base + tmp_5
    tmp_7 = torch.cat([prefix, tmp_6], dim=2)
    tmp_8 = tmp_7.type_as(ref_tensor)
    return tmp_8


def replacement_args(neg_input, slice_input, prefix, mul_base, sin_emb, ref_tensor):
    return (neg_input, slice_input, prefix, mul_base, sin_emb, ref_tensor)


@triton.jit
def fused_rope_vectorized_kernel(
    neg_ptr, slice_ptr, prefix_ptr, mul_base_ptr, sin_emb_ptr, output_ptr,
    batch, heads, seq_len, head_dim,
    half_dim,
    stride_nb, stride_nh, stride_ns, stride_nd,
    stride_sb, stride_sh, stride_ss, stride_sd,
    stride_pb, stride_ph, stride_ps, stride_pd,
    stride_mb, stride_mh, stride_ms, stride_md,
    stride_sem, stride_sed,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_main = batch * heads * seq_len * head_dim
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_main
    
    d = offsets % head_dim
    s = (offsets // head_dim) % seq_len
    h = (offsets // (head_dim * seq_len)) % heads
    b = offsets // (head_dim * seq_len * heads)
    
    mul_idx = b * stride_mb + h * stride_mh + s * stride_ms + d * stride_md
    mul_val = tl.load(mul_base_ptr + mul_idx, mask=mask)
    
    sin_idx = s * stride_sem + d * stride_sed
    sin_val = tl.load(sin_emb_ptr + sin_idx, mask=mask)
    
    half_d = d // 2
    is_even = (d % 2) == 0
    
    neg_idx = b * stride_nb + h * stride_nh + s * stride_ns + half_d * stride_nd
    neg_val = tl.load(neg_ptr + neg_idx, mask=mask & is_even)
    
    slice_idx_val = d - 1
    slice_idx = b * stride_sb + h * stride_sh + s * stride_ss + slice_idx_val * stride_sd
    slice_val = tl.load(slice_ptr + slice_idx, mask=mask & (~is_even))
    
    interleaved = tl.where(is_even, -neg_val, slice_val)
    val = mul_val + sin_val * interleaved
    
    out_idx = b * stride_ob + h * stride_oh + (s + 1) * stride_os + d * stride_od
    tl.store(output_ptr + out_idx, val, mask=mask)


@triton.jit
def copy_prefix_vectorized_kernel(
    prefix_ptr, output_ptr,
    batch, heads, head_dim,
    stride_pb, stride_ph, stride_ps, stride_pd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = batch * heads * head_dim
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    d = offsets % head_dim
    h = (offsets // head_dim) % heads
    b = offsets // (head_dim * heads)
    
    prefix_idx = b * stride_pb + h * stride_ph + d * stride_pd
    vals = tl.load(prefix_ptr + prefix_idx, mask=mask)
    
    out_idx = b * stride_ob + h * stride_oh + d * stride_od
    tl.store(output_ptr + out_idx, vals, mask=mask)


@torch.fx.wrap
def fused_rope(neg_input, slice_input, prefix, mul_base, sin_emb, ref_tensor):
    batch, heads, seq_len, half_dim = neg_input.shape
    head_dim = half_dim * 2
    
    output = torch.empty(batch, heads, seq_len + 1, head_dim, 
                        dtype=ref_tensor.dtype, device=neg_input.device)
    
    BLOCK_SIZE = 1024
    
    total_prefix = batch * heads * head_dim
    num_blocks_prefix = (total_prefix + BLOCK_SIZE - 1) // BLOCK_SIZE
    copy_prefix_vectorized_kernel[(num_blocks_prefix,)](
        prefix, output,
        batch, heads, head_dim,
        prefix.stride(0), prefix.stride(1), prefix.stride(2), prefix.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    total_main = batch * heads * seq_len * head_dim
    num_blocks_main = (total_main + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_rope_vectorized_kernel[(num_blocks_main,)](
        neg_input, slice_input, prefix, mul_base, sin_emb, output,
        batch, heads, seq_len, head_dim,
        half_dim,
        neg_input.stride(0), neg_input.stride(1), neg_input.stride(2), neg_input.stride(3),
        slice_input.stride(0), slice_input.stride(1), slice_input.stride(2), slice_input.stride(3),
        prefix.stride(0), prefix.stride(1), prefix.stride(2), prefix.stride(3),
        mul_base.stride(0), mul_base.stride(1), mul_base.stride(2), mul_base.stride(3),
        sin_emb.stride(0), sin_emb.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_rope