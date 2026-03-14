import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Full pattern matching both RoPE and RMSNorm computations.
    """
    # RoPE computation
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    
    # RMSNorm computation
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    
    return (tmp_6, tmp_17, tmp_7)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def rope_kernel(
    in_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE cos/sin kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Each element appears twice in concatenated result
    half_n = n_elements // 2
    input_offsets = offsets % half_n
    x = tl.load(in_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute cos and sin
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store results
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)


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
    """Fused RMSNorm kernel."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    
    # Load and normalize
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = x * rstd
    
    # Apply weight
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = normed * weight
    
    tl.store(out_ptr + row_start + cols, out, mask=mask)


@torch.fx.wrap
def fused_rope_rmsnorm(weight, freqs, x, eps=1e-6):
    """
    Fused RoPE and RMSNorm computation.
    Returns: (cos, rmsnorm_out, sin)
    """
    # RoPE computation
    out_shape = list(freqs.shape)
    out_shape[-1] *= 2
    
    cos_out = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)
    sin_out = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)
    
    n_elements = cos_out.numel()
    BLOCK_SIZE_ROPE = 1024
    grid_rope = ((n_elements + BLOCK_SIZE_ROPE - 1) // BLOCK_SIZE_ROPE,)
    
    rope_kernel[grid_rope](
        freqs,
        cos_out,
        sin_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE_ROPE,
    )
    
    # RMSNorm computation
    assert x.is_contiguous()
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    n_rows = x_2d.shape[0]
    hidden_size = x_2d.shape[1]
    
    rmsnorm_out = torch.empty_like(x_2d)
    
    BLOCK_SIZE_NORM = 2048
    grid_norm = (n_rows,)
    
    rmsnorm_kernel[grid_norm](
        x_2d,
        weight,
        rmsnorm_out,
        stride_row=hidden_size,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE_NORM,
        num_warps=4,
    )
    
    rmsnorm_out = rmsnorm_out.view(orig_shape)
    
    return cos_out, rmsnorm_out, sin_out


def replacement_func():
    return fused_rope_rmsnorm