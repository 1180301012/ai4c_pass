import torch
import triton
import triton.language as tl


def pattern(e1, e2, e3, e4, e5, e6, e7, e8, e9, weight, bias):
    s = e1 + e2
    s = s + e3
    s = s + e4
    s = s + e5
    s = s + e6
    s = s + e7
    s = s + e8
    s = s + e9
    out = torch.nn.functional.layer_norm(s, (768,), weight, bias, 1e-12)
    out = torch.nn.functional.dropout(out, 0.1, False, False)
    return out


def replacement_args(e1, e2, e3, e4, e5, e6, e7, e8, e9, weight, bias):
    return (e1, e2, e3, e4, e5, e6, e7, e8, e9, weight, bias)


@triton.jit
def fused_add9_ln_no_bc_kernel(
    e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr, e6_ptr, e7_ptr, e8_ptr, e9_ptr,
    w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """No-broadcast kernel: all inputs have the same number of rows."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    base = row * N + offsets

    # Load and accumulate
    s = tl.load(e1_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e2_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e3_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e4_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e5_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e6_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e7_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e8_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e9_ptr + base, mask=mask, other=0.0).to(tl.float32)

    # Layer norm
    mean = tl.sum(s, axis=0) / N
    diff = s - mean
    sum_sq = tl.sum(diff * diff, axis=0)
    var = (sum_sq - (BLOCK_SIZE - N) * mean * mean) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-12)
    normalized = diff * inv_std

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = normalized * w + b

    tl.store(out_ptr + base, result, mask=mask)


@triton.jit
def fused_add9_ln_bc_kernel(
    e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr, e6_ptr, e7_ptr, e8_ptr, e9_ptr,
    w_ptr, b_ptr, out_ptr,
    e1_mask, e2_mask, e3_mask, e4_mask, e5_mask, e6_mask, e7_mask, e8_mask, e9_mask,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Broadcast kernel using bitwise AND (row counts are powers of 2)."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    s = tl.load(e1_ptr + (row & e1_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e2_ptr + (row & e2_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e3_ptr + (row & e3_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e4_ptr + (row & e4_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e5_ptr + (row & e5_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e6_ptr + (row & e6_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e7_ptr + (row & e7_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e8_ptr + (row & e8_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)
    s += tl.load(e9_ptr + (row & e9_mask) * N + offsets, mask=mask, other=0.0).to(tl.float32)

    # Layer norm
    mean = tl.sum(s, axis=0) / N
    diff = s - mean
    sum_sq = tl.sum(diff * diff, axis=0)
    var = (sum_sq - (BLOCK_SIZE - N) * mean * mean) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-12)
    normalized = diff * inv_std

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = normalized * w + b

    tl.store(out_ptr + row * N + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add9_layernorm(e1, e2, e3, e4, e5, e6, e7, e8, e9, weight, bias):
    N = 768

    # Compute number of rows for each input
    e1_rows = e1.shape[0] * e1.shape[1]
    e2_rows = e2.shape[0] * e2.shape[1]
    e3_rows = e3.shape[0] * e3.shape[1]
    e4_rows = e4.shape[0] * e4.shape[1]
    e5_rows = e5.shape[0] * e5.shape[1]
    e6_rows = e6.shape[0] * e6.shape[1]
    e7_rows = e7.shape[0] * e7.shape[1]
    e8_rows = e8.shape[0] * e8.shape[1]
    e9_rows = e9.shape[0] * e9.shape[1]

    # Output shape from broadcast
    out_batch = max(e1.shape[0], e2.shape[0], e3.shape[0], e4.shape[0], e5.shape[0],
                    e6.shape[0], e7.shape[0], e8.shape[0], e9.shape[0])
    out_seq = max(e1.shape[1], e2.shape[1], e3.shape[1], e4.shape[1], e5.shape[1],
                  e6.shape[1], e7.shape[1], e8.shape[1], e9.shape[1])
    total_rows = out_batch * out_seq

    out = torch.empty(out_batch, out_seq, N, dtype=weight.dtype, device=e1.device)

    # Check if any input needs broadcasting
    has_broadcast = (e1_rows != total_rows or e2_rows != total_rows or
                     e3_rows != total_rows or e4_rows != total_rows or
                     e5_rows != total_rows or e6_rows != total_rows or
                     e7_rows != total_rows or e8_rows != total_rows or
                     e9_rows != total_rows)

    if has_broadcast:
        fused_add9_ln_bc_kernel[(total_rows,)](
            e1, e2, e3, e4, e5, e6, e7, e8, e9,
            weight, bias, out,
            e1_rows - 1, e2_rows - 1, e3_rows - 1, e4_rows - 1,
            e5_rows - 1, e6_rows - 1, e7_rows - 1, e8_rows - 1, e9_rows - 1,
            N=N,
            BLOCK_SIZE=1024,
            num_warps=8,
        )
    else:
        fused_add9_ln_no_bc_kernel[(total_rows,)](
            e1, e2, e3, e4, e5, e6, e7, e8, e9,
            weight, bias, out,
            N=N,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

    return out


def replacement_func():
    return fused_add9_layernorm