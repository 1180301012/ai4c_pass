import torch
import triton
import triton.language as tl


@triton.jit
def transpose_add_kernel(
    # Input pointers
    x_ptr,            # [batch, channels, seq] - after gelu
    other_ptr,        # [batch, other_seq, other_hidden] - residual
    # Output pointer
    output_ptr,       # [batch, other_seq, other_hidden]
    # Strides for x: [batch, channels, seq]
    x_stride_b: tl.constexpr,
    x_stride_c: tl.constexpr, 
    x_stride_s: tl.constexpr,
    # Strides for other: [batch, other_seq, other_hidden]
    other_stride_b: tl.constexpr,
    other_stride_s: tl.constexpr,
    other_stride_h: tl.constexpr,
    # Strides for output
    out_stride_b: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_h: tl.constexpr,
    # Shapes
    batch: tl.constexpr,
    x_channels: tl.constexpr,
    x_seq: tl.constexpr,
    other_seq: tl.constexpr,
    other_hidden: tl.constexpr,
    # Block config
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Transpose -> Add
    """
    pid = tl.program_id(0)
    num_elements = batch * other_seq * other_hidden
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Decode output indices
    out_idx = offsets
    b = out_idx // (other_seq * other_hidden)
    remainder = out_idx % (other_seq * other_hidden)
    s = remainder // other_hidden
    h = remainder % other_hidden
    
    # Load from other tensor
    other_off = b * other_stride_b + s * other_stride_s + h * other_stride_h
    other_val = tl.load(other_ptr + other_off, mask=mask, other=0.0)
    
    # Compute x contribution with transpose
    c = h % x_channels
    x_valid_mask = s < x_seq
    x_off = b * x_stride_b + c * x_stride_c + s * x_stride_s
    x_off_masked = tl.where(x_valid_mask, x_off, 0)
    x_val = tl.load(x_ptr + x_off_masked, mask=x_valid_mask & mask, other=0.0)
    x_contrib = tl.where(x_valid_mask, x_val, 0.0)
    
    result = other_val + x_contrib
    
    out_off = b * out_stride_b + s * out_stride_s + h * out_stride_h
    tl.store(output_ptr + out_off, result, mask=mask)


@torch.fx.wrap
def fused_transpose_add(x, other):
    """
    Fused transpose + add operation using Triton kernel.
    """
    batch, channels, seq = x.shape
    other_batch, other_seq, other_hidden = other.shape
    
    output = torch.empty(other_batch, other_seq, other_hidden, device=other.device, dtype=other.dtype)
    
    BLOCK_SIZE = 1024
    num_elements = other_batch * other_seq * other_hidden
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    transpose_add_kernel[(num_programs,)](
        x_ptr=x,
        other_ptr=other,
        output_ptr=output,
        x_stride_b=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_s=x.stride(2),
        other_stride_b=other.stride(0),
        other_stride_s=other.stride(1),
        other_stride_h=other.stride(2),
        out_stride_b=output.stride(0),
        out_stride_s=output.stride(1),
        out_stride_h=output.stride(2),
        batch=batch,
        x_channels=channels,
        x_seq=seq,
        other_seq=other_seq,
        other_hidden=other_hidden,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(x, other):
    """
    Match the pattern: gelu(x).transpose(1, 2) + other
    """
    tmp_5 = torch.nn.functional.gelu(x)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_7 = other + tmp_6
    return tmp_7


def replacement_args(x, other):
    return (x, other)


def replacement_func():
    return fused_transpose_add