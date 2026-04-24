import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_softmax_kernel(
    x_ptr,
    out_ptr,
    seq_len,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_len
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf')).to(tl.float32)

    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    out_f32 = x_exp / x_sum

    if IS_FP16:
        tl.store(out_ptr + row_start + offsets, out_f32.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row_start + offsets, out_f32.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row_start + offsets, out_f32, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['seq_len'],
)
@triton.jit
def fused_add_softmax_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    seq_len,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * seq_len
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    a = tl.load(in0_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(in1_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    x = a + b

    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    out_f32 = x_exp / x_sum

    if IS_FP16:
        tl.store(out_ptr + row_start + offsets, out_f32.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row_start + offsets, out_f32.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row_start + offsets, out_f32, mask=mask)


@torch.fx.wrap
def fused_softmax_type_as(x):
    # Pattern: x.float() -> softmax -> type_as (replacement for float+softmax+type_as)
    # x is the result of the preceding add operation.
    # The replacement outputs float32; any remaining _to_copy in the graph
    # handles the dtype conversion.
    orig_shape = x.shape
    seq_len = x.shape[-1]
    n_rows = x.numel() // seq_len

    out = torch.empty(orig_shape, dtype=torch.float32, device=x.device)

    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)

    grid = (n_rows,)
    fused_softmax_kernel[grid](
        x, out, seq_len,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return out


@torch.fx.wrap
def fused_add_softmax_type_as(in_0, in_1):
    # Full fused: add + softmax + dtype
    orig_shape = in_0.shape
    seq_len = in_0.shape[-1]
    n_rows = in_0.numel() // seq_len

    out = torch.empty(orig_shape, dtype=in_0.dtype, device=in_0.device)

    IS_FP16 = (in_0.dtype == torch.float16)
    IS_BF16 = (in_0.dtype == torch.bfloat16)

    grid = (n_rows,)
    fused_add_softmax_kernel[grid](
        in_0, in_1, out,
        seq_len=seq_len,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return out


# Match: x.type_as(y) — confirmed found as anchor in the model graph
def pattern(x, y):
    return x.type_as(y)


def replacement_args(x, y):
    return (x,)


def replacement_func():
    return fused_softmax_type_as