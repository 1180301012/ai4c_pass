import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match: add + layer_norm
    """
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (256,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_layer_norm_kernel(
    in_2_ptr,
    in_3_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride,
    N,  # Normalized dimension size (256)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: add + layer_norm
    Each program handles one row using efficient single-pass algorithm
    """
    row_idx = tl.program_id(0)
    
    # Compute base offset for this row
    row_start = row_idx * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    offsets = row_start + col_offsets
    
    # Load and fuse add
    in_2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    x = in_2 + in_3
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance (using masked values properly)
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    out = xm * rstd
    
    # Apply affine parameters
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    out = out * weight + bias_val
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layer_norm(in_0, in_1, in_2, in_3):
    """
    Wrapper function for the fused kernel
    in_0: bias [256]
    in_1: weight [256]
    in_2, in_3: input tensors [batch, seq, 256]
    """
    # Flatten for simpler indexing
    batch, seq_len, hidden_dim = in_2.shape
    M = batch * seq_len
    N = hidden_dim
    
    # Allocate output
    output = torch.empty_like(in_2)
    
    # Launch kernel with one thread block per sequence element
    grid = (M,)
    BLOCK_SIZE = 256  # Match the hidden dimension
    
    fused_add_layer_norm_kernel[grid](
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=output,
        stride=N,
        N=N,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_add_layer_norm