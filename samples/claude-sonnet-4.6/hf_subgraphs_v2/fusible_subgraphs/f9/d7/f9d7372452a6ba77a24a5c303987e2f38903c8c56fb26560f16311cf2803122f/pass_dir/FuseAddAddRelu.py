import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Pattern: match the in-place add + in-place add + relu(inplace) sequence
# --------------------------------------------------------------------------
def pattern(in_0, in_2, in_3):
    in_3 += in_0
    in_3 += in_2
    tmp_2 = torch.nn.functional.relu(in_3, inplace=True)
    return tmp_2


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


# --------------------------------------------------------------------------
# Triton kernel: fused  out = relu(a + b + c)
# --------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_add_add_relu_kernel(
    a_ptr,       # in_3 pointer
    b_ptr,       # in_0 pointer
    c_ptr,       # in_2 pointer
    out_ptr,     # output pointer
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)

    val = a + b + c
    out = tl.where(val > 0.0, val, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


# --------------------------------------------------------------------------
# Kernel wrapper
# --------------------------------------------------------------------------
@torch.fx.wrap
def fused_add_add_relu(in_0, in_2, in_3):
    n = in_3.numel()
    out = torch.empty_like(in_3)
    # Grid is computed inside the autotuned launcher; pass a lambda
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _fused_add_add_relu_kernel[grid](
        in_3, in_0, in_2, out, n
    )
    return out


# --------------------------------------------------------------------------
# Replacement factory
# --------------------------------------------------------------------------
def replacement_func():
    return fused_add_add_relu