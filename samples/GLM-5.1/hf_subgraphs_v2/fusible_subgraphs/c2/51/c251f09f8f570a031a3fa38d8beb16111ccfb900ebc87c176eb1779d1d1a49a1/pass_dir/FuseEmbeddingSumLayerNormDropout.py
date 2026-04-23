import torch
import triton
import triton.language as tl


def pattern(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    t1 = e0 + e1
    t2 = t1 + e2
    t3 = t2 + e3
    t4 = t3 + e4
    t5 = t4 + e5
    t6 = t5 + e6
    t7 = t6 + e7
    t8 = t7 + e8
    t9 = torch.nn.functional.layer_norm(t8, (768,), ln_weight, ln_bias, 1e-12)
    t10 = torch.nn.functional.dropout(t9, 0.1, False, False)
    return t10


def replacement_args(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    return (e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias)


@triton.jit
def fused_sum_layernorm_kernel(
    e0_ptr, e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr, e6_ptr, e7_ptr, e8_ptr,
    weight_ptr, bias_ptr, out_ptr,
    n_batch, n_seq,
    HIDDEN_SIZE,
    eps,
    stride_batch_e0, stride_seq_e0,
    stride_batch_e1, stride_seq_e1,
    stride_batch_e2, stride_seq_e2,
    stride_batch_e3, stride_seq_e3,
    stride_batch_e4, stride_seq_e4,
    stride_batch_e5, stride_seq_e5,
    stride_batch_e6, stride_seq_e6,
    stride_batch_e7, stride_seq_e7,
    stride_batch_e8, stride_seq_e8,
    stride_batch_out, stride_seq_out,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HIDDEN_SIZE

    # Compute offsets for each embedding (stride_batch=0 means broadcast)
    e0_row = e0_ptr + batch_idx * stride_batch_e0 + seq_idx * stride_seq_e0 + cols
    e1_row = e1_ptr + batch_idx * stride_batch_e1 + seq_idx * stride_seq_e1 + cols
    e2_row = e2_ptr + batch_idx * stride_batch_e2 + seq_idx * stride_seq_e2 + cols
    e3_row = e3_ptr + batch_idx * stride_batch_e3 + seq_idx * stride_seq_e3 + cols
    e4_row = e4_ptr + batch_idx * stride_batch_e4 + seq_idx * stride_seq_e4 + cols
    e5_row = e5_ptr + batch_idx * stride_batch_e5 + seq_idx * stride_seq_e5 + cols
    e6_row = e6_ptr + batch_idx * stride_batch_e6 + seq_idx * stride_seq_e6 + cols
    e7_row = e7_ptr + batch_idx * stride_batch_e7 + seq_idx * stride_seq_e7 + cols
    e8_row = e8_ptr + batch_idx * stride_batch_e8 + seq_idx * stride_seq_e8 + cols

    # Load and accumulate all embeddings (upcast to float32 for numerical stability)
    x = tl.load(e0_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e1_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e2_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e3_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e4_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e5_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e6_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e7_row, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(e8_row, mask=mask, other=0.0).to(tl.float32)

    # Mask out padding elements for correct layer norm statistics
    x_masked = tl.where(mask, x, 0.0)

    # Layer norm computation in float32 (only over valid HIDDEN_SIZE elements)
    mean = tl.sum(x_masked, axis=0) / HIDDEN_SIZE
    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / HIDDEN_SIZE
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Load weight and bias (upcast to float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b_ln = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Apply layer norm: (x - mean) / sqrt(var + eps) * weight + bias (only for valid elements)
    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
    out = x_hat * w + b_ln

    # Store result
    out_row = out_ptr + batch_idx * stride_batch_out + seq_idx * stride_seq_out + cols
    tl.store(out_row, out, mask=mask)


@torch.fx.wrap
def fused_sum_layernorm_dropout(e0, e1, e2, e3, e4, e5, e6, e7, e8, ln_weight, ln_bias):
    # Get shapes using .size() - allowed by PosionDispatchTensor
    # Find max batch size (for broadcasting)
    bs0 = e0.size()[0]
    bs1 = e1.size()[0]
    bs2 = e2.size()[0]
    bs3 = e3.size()[0]
    bs4 = e4.size()[0]
    bs5 = e5.size()[0]
    bs6 = e6.size()[0]
    bs7 = e7.size()[0]
    bs8 = e8.size()[0]
    batch_size = max(bs0, bs1, bs2, bs3, bs4, bs5, bs6, bs7, bs8)
    
    seq_len = e0.size()[1]
    hidden_size = e0.size()[2]

    # Allocate output - torch.empty is allowed by PosionDispatchTensor
    # Use e0.dtype and e0.device (these are properties, not __torch_dispatch__ operations)
    out = torch.empty(batch_size, seq_len, hidden_size, dtype=e0.dtype, device=e0.device)

    # Get strides - .stride() is allowed by PosionDispatchTensor
    # For broadcasting: if an input has batch_size < output batch_size, use stride[0]=0
    e0_strides = e0.stride()
    e1_strides = e1.stride()
    e2_strides = e2.stride()
    e3_strides = e3.stride()
    e4_strides = e4.stride()
    e5_strides = e5.stride()
    e6_strides = e6.stride()
    e7_strides = e7.stride()
    e8_strides = e8.stride()
    out_strides = out.stride()

    # Handle broadcasting: set stride[0] to 0 if the input has batch_size=1 but output needs more
    s0 = e0_strides[0] if bs0 == batch_size else 0
    s1 = e1_strides[0] if bs1 == batch_size else 0
    s2 = e2_strides[0] if bs2 == batch_size else 0
    s3 = e3_strides[0] if bs3 == batch_size else 0
    s4 = e4_strides[0] if bs4 == batch_size else 0
    s5 = e5_strides[0] if bs5 == batch_size else 0
    s6 = e6_strides[0] if bs6 == batch_size else 0
    s7 = e7_strides[0] if bs7 == batch_size else 0
    s8 = e8_strides[0] if bs8 == batch_size else 0

    # Launch kernel with 2D grid
    grid = (batch_size, seq_len)
    fused_sum_layernorm_kernel[grid](
        e0, e1, e2, e3, e4, e5, e6, e7, e8,
        ln_weight, ln_bias, out,
        batch_size, seq_len,
        hidden_size,
        1e-12,
        s0, e0_strides[1],
        s1, e1_strides[1],
        s2, e2_strides[1],
        s3, e3_strides[1],
        s4, e4_strides[1],
        s5, e5_strides[1],
        s6, e6_strides[1],
        s7, e7_strides[1],
        s8, e8_strides[1],
        out_strides[0], out_strides[1],
        BLOCK_SIZE=1024,
    )

    return out


def replacement_func():
    return fused_sum_layernorm_dropout