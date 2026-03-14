import torch
import triton
import triton.language as tl


def pattern(in_0, in_2):
    """
    Pattern for RMSNorm computation with weight multiplication:
    x -> to(float32) -> pow(2) -> mean -> add eps -> rsqrt -> mul -> to(bf16) -> mul weight
    """
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_0, in_2):
    return (in_0, in_2)


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm kernel with weight multiplication.
    Processes one row (last dimension) per program.
    """
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate row start pointer
    row_start = row_idx * stride_row
    
    # Load the entire row and compute variance
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    
    # Load input data
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute RMS
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    normed = x * rstd
    
    # Load weight and multiply
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = normed * weight
    
    # Store result
    tl.store(out_ptr + row_start + cols, out, mask=mask)


@torch.fx.wrap
def fused_rmsnorm(weight, x, eps=1e-6):
    """
    Fused RMSNorm with weight multiplication.
    Input: x [B, S, H] bfloat16, weight [H] bfloat16
    Output: [B, S, H] bfloat16
    """
    assert x.is_contiguous()
    
    # Flatten to 2D for processing
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    n_rows = x_2d.shape[0]
    hidden_size = x_2d.shape[1]
    
    out = torch.empty_like(x_2d)
    
    # Use fixed BLOCK_SIZE optimized for hidden_size=2048
    BLOCK_SIZE = 2048
    
    grid = (n_rows,)
    
    rmsnorm_kernel[grid](
        x_2d,
        weight,
        out,
        stride_row=hidden_size,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return out.view(orig_shape)


def replacement_func():
    return fused_rmsnorm