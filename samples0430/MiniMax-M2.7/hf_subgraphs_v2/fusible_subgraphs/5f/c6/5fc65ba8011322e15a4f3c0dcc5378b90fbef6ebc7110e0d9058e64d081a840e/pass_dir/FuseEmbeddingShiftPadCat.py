import torch
import triton
import triton.language as tl


@triton.jit
def triton_cat_dim2_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    a_batch_stride, a_seq_stride, a_dim_stride,
    b_batch_stride, b_seq_stride, b_dim_stride,
    c_batch_stride, c_seq_stride, c_dim_stride,
    output_batch_stride, output_seq_stride, output_dim_stride,
    batch_size, seq_len, part_size: tl.constexpr
):
    """
    Triton kernel for concatenating 3 tensors along dim 2.
    Each input has shape [batch, seq, part_size].
    Output has shape [batch, seq, 3*part_size].
    """
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # For input a: output[:, :, 0:part_size]
    a_offset = batch_idx * a_batch_stride + seq_idx * a_seq_stride
    out_offset_a = batch_idx * output_batch_stride + seq_idx * output_dim_stride
    for dim_idx in range(part_size):
        val = tl.load(a_ptr + a_offset + dim_idx * a_dim_stride)
        tl.store(output_ptr + out_offset_a + dim_idx, val)
    
    # For input b: output[:, :, part_size:2*part_size]
    b_offset = batch_idx * b_batch_stride + seq_idx * b_seq_stride
    out_offset_b = out_offset_a + part_size
    for dim_idx in range(part_size):
        val = tl.load(b_ptr + b_offset + dim_idx * b_dim_stride)
        tl.store(output_ptr + out_offset_b + dim_idx, val)
    
    # For input c: output[:, :, 2*part_size:3*part_size]
    c_offset = batch_idx * c_batch_stride + seq_idx * c_seq_stride
    out_offset_c = out_offset_a + 2 * part_size
    for dim_idx in range(part_size):
        val = tl.load(c_ptr + c_offset + dim_idx * c_dim_stride)
        tl.store(output_ptr + out_offset_c + dim_idx, val)


@torch.fx.wrap
def fused_cat_3(a, b, c):
    """
    Triton implementation of concatenating 3 tensors along dim 2.
    """
    batch_size, seq_len, part_size = a.shape
    
    output = torch.empty(
        (batch_size, seq_len, 3 * part_size),
        dtype=a.dtype,
        device=a.device
    )
    
    grid = (batch_size, seq_len)
    
    triton_cat_dim2_kernel[grid](
        a, b, c, output,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        batch_size, seq_len, part_size
    )
    
    return output


def pattern(a, b, c):
    return torch.cat([a, b, c], dim=2)


def replacement_args(a, b, c):
    return (a, b, c)


def replacement_func():
    return fused_cat_3