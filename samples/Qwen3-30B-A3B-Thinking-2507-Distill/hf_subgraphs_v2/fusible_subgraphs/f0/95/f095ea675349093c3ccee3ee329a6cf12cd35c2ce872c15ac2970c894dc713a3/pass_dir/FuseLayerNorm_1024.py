import torch
import triton
import triton.language as tl


def pattern(x, in_1, in_0):
    """
    Matches the layer_norm applied to tmp_8 (output of add/dropout).
    Single output - avoids multi-output tuple issues.
    """
    return torch.nn.functional.layer_norm(x, (1024,), in_1, in_0, 1e-05)


def replacement_args(x, in_1, in_0):
    return (x, in_1, in_0)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def _layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row.
    x layout [B, S, C=N]: row-major, element [b,s,c] at row*N + c.
    """
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    x_f32 = tl.load(x_ptr + row * N + offs).to(tl.float32)

    # mean
    mean    = tl.sum(x_f32, axis=0) / BLOCK_SIZE
    diff    = x_f32 - mean
    # variance
    var     = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    x_norm  = diff * inv_std

    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (x_norm * w + b).to(x_f32.dtype))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def _layernorm_bf16_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    x_f32 = tl.load(x_ptr + row * N + offs).to(tl.float32)

    mean    = tl.sum(x_f32, axis=0) / BLOCK_SIZE
    diff    = x_f32 - mean
    var     = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    x_norm  = diff * inv_std

    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (x_norm * w + b).to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def _layernorm_fp16_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    x_f32 = tl.load(x_ptr + row * N + offs).to(tl.float32)

    mean    = tl.sum(x_f32, axis=0) / BLOCK_SIZE
    diff    = x_f32 - mean
    var     = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    x_norm  = diff * inv_std

    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)
    tl.store(out_ptr + row * N + offs, (x_norm * w + b).to(tl.float16))


@torch.fx.wrap
def triton_layer_norm(x, in_1, in_0):
    """
    Drop-in replacement for torch.nn.functional.layer_norm(x, (1024,), in_1, in_0, 1e-05).
    x : [B, S, C=1024]
    """
    B, S, C = x.shape
    N = C
    rows = B * S
    out = torch.empty_like(x)

    if x.dtype == torch.bfloat16:
        _layernorm_bf16_kernel[(rows,)](x, in_1, in_0, out, N, BLOCK_SIZE=1024)
    elif x.dtype == torch.float16:
        _layernorm_fp16_kernel[(rows,)](x, in_1, in_0, out, N, BLOCK_SIZE=1024)
    else:
        _layernorm_kernel[(rows,)](x, in_1, in_0, out, N, BLOCK_SIZE=1024)

    return out


def replacement_func():
    return triton_layer_norm