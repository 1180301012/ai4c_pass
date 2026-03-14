import torch
import triton
import triton.language as tl

# Pattern to match: entire computation including rotary embedding and layer_norm
def pattern(bias, weight, freqs, x):
    tmp_2 = torch.cat((freqs, freqs), dim=-1)
    tmp_3 = tmp_2.cos()
    tmp_4 = tmp_3 * 1.0
    tmp_5 = tmp_2.sin()
    tmp_6 = tmp_5 * 1.0
    tmp_7 = tmp_4.to(dtype=torch.float16)
    tmp_8 = tmp_6.to(dtype=torch.float16)
    tmp_11 = torch.nn.functional.layer_norm(x, (2560,), weight, bias, 1e-05)
    return (tmp_7, tmp_11, tmp_8)

def replacement_args(bias, weight, freqs, x):
    return (bias, weight, freqs, x)


# Process multiple rows of rotary embedding per block for better efficiency
@triton.jit
def rotary_embed_batched_kernel(
    freqs_ptr,
    cos_ptr,
    sin_ptr,
    num_rows,
    in_dim,
    out_dim,
    ROWS_PER_BLOCK: tl.constexpr,
    IN_DIM: tl.constexpr,
):
    block_idx = tl.program_id(0)
    row_start = block_idx * ROWS_PER_BLOCK
    
    in_offsets = tl.arange(0, IN_DIM)
    
    # Process ROWS_PER_BLOCK rows with static unrolling
    for row_offset in tl.static_range(ROWS_PER_BLOCK):
        row_idx = row_start + row_offset
        if row_idx < num_rows:
            in_base = row_idx * in_dim
            
            freqs = tl.load(freqs_ptr + in_base + in_offsets)
            cos_val = tl.cos(freqs).to(tl.float16)
            sin_val = tl.sin(freqs).to(tl.float16)
            
            out_base = row_idx * out_dim
            tl.store(cos_ptr + out_base + in_offsets, cos_val)
            tl.store(cos_ptr + out_base + in_dim + in_offsets, cos_val)
            tl.store(sin_ptr + out_base + in_offsets, sin_val)
            tl.store(sin_ptr + out_base + in_dim + in_offsets, sin_val)


# Layer norm with fixed optimal config
@triton.jit
def layer_norm_kernel_v2(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * N
    out_row_ptr = out_ptr + row_idx * N
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load x
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Mean
    x_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = x_sum / N
    
    # Variance
    x_centered = x - mean
    var_sum = tl.sum(tl.where(mask, x_centered * x_centered, 0.0), axis=0)
    var = var_sum / N
    rstd = tl.rsqrt(var + eps)
    
    # Normalize and apply affine
    x_hat = x_centered * rstd
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_hat * w + b
    
    tl.store(out_row_ptr + cols, y.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_computation_impl(bias, weight, freqs, x):
    # Rotary embedding
    batch_size, seq_len, in_dim = freqs.shape
    out_dim = in_dim * 2
    num_rows = batch_size * seq_len
    
    cos_out = torch.empty(batch_size, seq_len, out_dim, dtype=torch.float16, device=freqs.device)
    sin_out = torch.empty(batch_size, seq_len, out_dim, dtype=torch.float16, device=freqs.device)
    
    ROWS_PER_BLOCK_ROTARY = 16
    num_blocks_rotary = (num_rows + ROWS_PER_BLOCK_ROTARY - 1) // ROWS_PER_BLOCK_ROTARY
    
    rotary_embed_batched_kernel[(num_blocks_rotary,)](
        freqs,
        cos_out,
        sin_out,
        num_rows,
        in_dim,
        out_dim,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK_ROTARY,
        IN_DIM=16,
        num_warps=1,
    )
    
    # Layer norm
    original_shape = x.shape
    N = original_shape[-1]
    x_2d = x.view(-1, N)
    num_rows_ln = x_2d.shape[0]
    
    ln_out = torch.empty_like(x_2d)
    
    layer_norm_kernel_v2[(num_rows_ln,)](
        x_2d,
        weight,
        bias,
        ln_out,
        N,
        1e-05,
        BLOCK_SIZE=4096,
        num_warps=8,
    )
    
    return (cos_out, ln_out.view(original_shape), sin_out)


def fused_computation(bias, weight, freqs, x):
    result = fused_computation_impl(bias, weight, freqs, x)
    return result[0], result[1], result[2]


def replacement_func():
    return fused_computation