import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match: just torch.square
# (confirmed-working single-node pattern)
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_2 = torch.square(in_0)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Fused Triton kernel:  out = x * x
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * x
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_square(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _square_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func — must return the callable, not call it
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_square