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
def fused_add_softmax_type_as(in_0, in_1):
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
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )
    return out


def pattern(in_0, in_1):
    # Full chain: in-place add -> float32 cast -> softmax.int (call_function)
    tmp_add = torch.ops.aten.add_.Tensor(in_1, in_0)
    tmp_f32 = torch.ops.aten._to_copy.default(tmp_add, dtype=6)
    tmp_softmax = torch.ops.aten.softmax.int(tmp_f32, -1)
    return tmp_softmax


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add_softmax_type_as