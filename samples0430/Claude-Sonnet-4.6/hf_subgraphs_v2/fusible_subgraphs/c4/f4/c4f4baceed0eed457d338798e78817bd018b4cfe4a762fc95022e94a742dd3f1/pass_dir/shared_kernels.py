import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Roll+Add kernels  — stride-aware, no .view()/.contiguous() needed
# in_3 shape: (1, n_h, 7, n_w, 7, C);  strides s1..s5 for dims 1..5
# in_2 shape: (1, N, C);  strides s2_n, s2_c for dims 1,2
# out:  torch.empty([1, N, C]) → contiguous, stride (N*C, C, 1)
# ---------------------------------------------------------------------------

@triton.jit
def _roll_add_35_384(
    in3_ptr, s1, s2, s3, s4, s5,
    in2_ptr, s2_n, s2_c,
    out_ptr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    row = n >> 5;  col = n & 31
    sr = (row + 32) % 35;  sc = (col + 32) % 35
    i = sr // 7;  j = sr % 7
    k = sc // 7;  l = sc % 7
    base3 = i * s1 + j * s2 + k * s3 + l * s4
    c = tl.arange(0, BLOCK_C);  mask = c < 384
    x = tl.load(in3_ptr + base3 + c * s5, mask=mask, other=0.0)
    y = tl.load(in2_ptr + n * s2_n + c * s2_c, mask=mask, other=0.0)
    tl.store(out_ptr + n * 384 + c, x + y, mask=mask)


@triton.jit
def _roll_add_70_192(
    in3_ptr, s1, s2, s3, s4, s5,
    in2_ptr, s2_n, s2_c,
    out_ptr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    row = n >> 6;  col = n & 63
    sr = (row + 67) % 70;  sc = (col + 67) % 70
    i = sr // 7;  j = sr % 7
    k = sc // 7;  l = sc % 7
    base3 = i * s1 + j * s2 + k * s3 + l * s4
    c = tl.arange(0, BLOCK_C);  mask = c < 192
    x = tl.load(in3_ptr + base3 + c * s5, mask=mask, other=0.0)
    y = tl.load(in2_ptr + n * s2_n + c * s2_c, mask=mask, other=0.0)
    tl.store(out_ptr + n * 192 + c, x + y, mask=mask)


@triton.jit
def _roll_add_133_96(
    in3_ptr, s1, s2, s3, s4, s5,
    in2_ptr, s2_n, s2_c,
    out_ptr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    row = n >> 7;  col = n & 127
    sr = (row + 130) % 133;  sc = (col + 130) % 133
    i = sr // 7;  j = sr % 7
    k = sc // 7;  l = sc % 7
    base3 = i * s1 + j * s2 + k * s3 + l * s4
    c = tl.arange(0, BLOCK_C);  mask = c < 96
    x = tl.load(in3_ptr + base3 + c * s5, mask=mask, other=0.0)
    y = tl.load(in2_ptr + n * s2_n + c * s2_c, mask=mask, other=0.0)
    tl.store(out_ptr + n * 96 + c, x + y, mask=mask)


# ---------------------------------------------------------------------------
# Layer-norm kernels — stride-aware
# in_2 shape: (1, N, C);  strides s_n, s_c
# weight in_1 (C,); bias in_0 (C,) — assumed contiguous (stride 1)
# out: torch.empty([1, N, C]) → contiguous
# ---------------------------------------------------------------------------

@triton.jit
def _layer_norm_384(
    in_ptr, s_n, s_c,
    w_ptr, b_ptr,
    out_ptr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.arange(0, BLOCK_C);  mask = c < 384
    x = tl.load(in_ptr + n * s_n + c * s_c, mask=mask, other=0.0).to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, 0) / 384.0
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, 0) / 384.0
    rstd = tl.rsqrt(var + 1e-5)
    w = tl.load(w_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + n * 384 + c, (x - mean) * rstd * w + b, mask=mask)


@triton.jit
def _layer_norm_192(
    in_ptr, s_n, s_c,
    w_ptr, b_ptr,
    out_ptr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.arange(0, BLOCK_C);  mask = c < 192
    x = tl.load(in_ptr + n * s_n + c * s_c, mask=mask, other=0.0).to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, 0) / 192.0
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, 0) / 192.0
    rstd = tl.rsqrt(var + 1e-5)
    w = tl.load(w_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + n * 192 + c, (x - mean) * rstd * w + b, mask=mask)


@triton.jit
def _layer_norm_96(
    in_ptr, s_n, s_c,
    w_ptr, b_ptr,
    out_ptr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.arange(0, BLOCK_C);  mask = c < 96
    x = tl.load(in_ptr + n * s_n + c * s_c, mask=mask, other=0.0).to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, 0) / 96.0
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, 0) / 96.0
    rstd = tl.rsqrt(var + 1e-5)
    w = tl.load(w_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + n * 96 + c, (x - mean) * rstd * w + b, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers — only torch.empty for allocation; .stride() for metadata
# ---------------------------------------------------------------------------

def _fast_roll_add_35_384(in_2, in_3):
    N, C = 1024, 384
    out = torch.empty([1, N, C], dtype=in_2.dtype, device=in_2.device)
    s3 = in_3.stride()
    s2 = in_2.stride()
    _roll_add_35_384[(N,)](
        in_3, s3[1], s3[2], s3[3], s3[4], s3[5],
        in_2, s2[1], s2[2],
        out, BLOCK_C=512,
    )
    return out


def _fast_roll_add_70_192(in_2, in_3):
    N, C = 4096, 192
    out = torch.empty([1, N, C], dtype=in_2.dtype, device=in_2.device)
    s3 = in_3.stride()
    s2 = in_2.stride()
    _roll_add_70_192[(N,)](
        in_3, s3[1], s3[2], s3[3], s3[4], s3[5],
        in_2, s2[1], s2[2],
        out, BLOCK_C=256,
    )
    return out


def _fast_roll_add_133_96(in_2, in_3):
    N, C = 16384, 96
    out = torch.empty([1, N, C], dtype=in_2.dtype, device=in_2.device)
    s3 = in_3.stride()
    s2 = in_2.stride()
    _roll_add_133_96[(N,)](
        in_3, s3[1], s3[2], s3[3], s3[4], s3[5],
        in_2, s2[1], s2[2],
        out, BLOCK_C=128,
    )
    return out


def _fast_ln_384(in_0, in_1, in_2):
    """in_0=bias(384,), in_1=weight(384,), in_2=input(1,N,384)"""
    N, C = 1024, 384
    out = torch.empty([1, N, C], dtype=in_2.dtype, device=in_2.device)
    s = in_2.stride()
    _layer_norm_384[(N,)](in_2, s[1], s[2], in_1, in_0, out, BLOCK_C=512)
    return out


def _fast_ln_192(in_0, in_1, in_2):
    N, C = 4096, 192
    out = torch.empty([1, N, C], dtype=in_2.dtype, device=in_2.device)
    s = in_2.stride()
    _layer_norm_192[(N,)](in_2, s[1], s[2], in_1, in_0, out, BLOCK_C=256)
    return out


def _fast_ln_96(in_0, in_1, in_2):
    N, C = 16384, 96
    out = torch.empty([1, N, C], dtype=in_2.dtype, device=in_2.device)
    s = in_2.stride()
    _layer_norm_96[(N,)](in_2, s[1], s[2], in_1, in_0, out, BLOCK_C=128)
    return out


# ---------------------------------------------------------------------------
# Shared dispatch  (all passes return this object → single replacement_func)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def shared_dispatch(a, b, c, route):
    if route == 'r35_384':
        return _fast_roll_add_35_384(a, b)
    elif route == 'r70_192':
        return _fast_roll_add_70_192(a, b)
    elif route == 'r133_96':
        return _fast_roll_add_133_96(a, b)
    elif route == 'ln384':
        return _fast_ln_384(a, b, c)
    elif route == 'ln192':
        return _fast_ln_192(a, b, c)
    else:  # ln96
        return _fast_ln_96(a, b, c)