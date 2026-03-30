"""
Pass 3: FuseAddDropoutEval

Key insight:
  torch.nn.functional.dropout(x, p, training=False, ...) is an identity in eval mode.
  So: tmp_24 = dropout(tmp_12 + tmp_22, 0.1, False, False)
      is equivalent to: tmp_24 = tmp_12 + tmp_22

  Replace with a Triton fused element-wise add kernel, skipping the dropout kernel launch.
"""

import torch
import triton
import triton.language as tl


def pattern(a, b):
    tmp = a + b
    out = torch.nn.functional.dropout(tmp, 0.1, False, False)
    return out


def replacement_args(a, b):
    return (a, b)


@triton.jit
def fused_add_eval_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@torch.fx.wrap
def fused_add_eval(a, b):
    # a, b: same shape (e.g. [1, 236, 32] = 7552 elements), contiguous
    a_c = a.contiguous()
    b_c = b.contiguous()
    n_elements = a_c.numel()
    out = torch.empty_like(a_c)
    BLOCK_SIZE = 256
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_add_eval_kernel[(num_programs,)](
        a_ptr=a_c,
        b_ptr=b_c,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_add_eval