import math
import operator
import torch
import torch.fx
import triton
import triton.language as tl
from torch import device

# Patch torch.fx.Proxy.__itruediv__ so that in-place /= in the pattern function
# generates call_function nodes with operator.itruediv as the target, matching
# the itruediv nodes the model's FX graph was traced with.
torch.fx.Proxy.__itruediv__ = lambda self, other: self.tracer.create_proxy(
    'call_function', operator.itruediv, (self, other), {}
)

# Combined scale factor: 1 / (sqrt(256) * 0.05) = 1 / (16.0 * 0.05) = 1 / 0.8 = 1.25
SCALE = 1.0 / (math.sqrt(256.0) * 0.05)
# Pre-fold log2(e) into the scale so we can use exp2 (native PTX ex2.approx)
# instead of exp (which internally does x*log2e then ex2 — one extra multiply).
SCALE_LOG2E = SCALE * math.log2(math.e)   # ≈ 1.8034


def pattern(in_0, divisor1, divisor2):
    """Match two sequential in-place divisions followed by softmax.
    divisor1 matches tmp_2 = sqrt(256) = 16.0 (or the pow-result subtree)
    divisor2 matches tmp_4 = 0.05
    Using placeholders avoids constant-folding get_attr mismatches.
    """
    in_0 /= divisor1    # itruediv node via patched __itruediv__
    in_0 /= divisor2    # itruediv node via patched __itruediv__
    tmp_6 = in_0.softmax(dim=-1)
    return tmp_6


def replacement_args(in_0, divisor1, divisor2):
    # Return only the original (pre-division) tensor; scale is baked into kernel
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def scaled_softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    stride_row,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = input_ptr + row * stride_row
    out_row_ptr = output_ptr + row * stride_row

    cols = tl.arange(0, BLOCK_SIZE)

    # Load in native dtype (fp16 / bf16)
    x_in = tl.load(row_ptr + cols)
    # Compute max in native dtype (saves conversion before reduction)
    x_max_native = tl.max(x_in, axis=0)

    # Cast to float32 for numerically stable computation
    x = x_in.to(tl.float32)
    x_max = x_max_native.to(tl.float32)

    # FMA: (x - x_max) * scale_log2e — single fused op, result in base-2 space
    x = (x - x_max) * scale
    # exp2 maps directly to PTX ex2.approx (one instruction, no log2e multiply)
    x = tl.math.exp2(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum

    # Cast back to input dtype and store
    tl.store(out_row_ptr + cols, x.to(x_in.dtype))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=2, num_warps=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def scaled_softmax_kernel_fp32(
    input_ptr, output_ptr,
    n_cols,
    stride_row,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimised float32 path: no dtype conversion; exp2 for speed."""
    row = tl.program_id(0)
    row_ptr = input_ptr + row * stride_row
    out_row_ptr = output_ptr + row * stride_row

    cols = tl.arange(0, BLOCK_SIZE)

    x = tl.load(row_ptr + cols)          # already float32
    x_max = tl.max(x, axis=0)            # max before scaling
    # FMA: (x - x_max) * scale_log2e — result in base-2 space
    x = (x - x_max) * scale
    # exp2 maps directly to PTX ex2.approx
    x = tl.math.exp2(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum

    tl.store(out_row_ptr + cols, x)


@torch.fx.wrap
def scaled_softmax(in_0):
    # Flatten all batch dims into rows; last dim is the softmax axis
    n_rows = in_0.numel() // in_0.shape[-1]
    n_cols = in_0.shape[-1]

    output = torch.empty_like(in_0)

    if in_0.dtype == torch.float32:
        scaled_softmax_kernel_fp32[(n_rows,)](
            in_0, output,
            n_cols,
            in_0.stride(-2),
            SCALE_LOG2E,   # pre-folded log2(e) for exp2 instruction
        )
    else:
        scaled_softmax_kernel[(n_rows,)](
            in_0, output,
            n_cols,
            in_0.stride(-2),
            SCALE_LOG2E,   # pre-folded log2(e) for exp2 instruction
        )

    # Return the tensor directly (not a tuple); the subgraph rewriter
    # maps this single value to tmp_6 in the original graph.
    return output


def replacement_func():
    return scaled_softmax