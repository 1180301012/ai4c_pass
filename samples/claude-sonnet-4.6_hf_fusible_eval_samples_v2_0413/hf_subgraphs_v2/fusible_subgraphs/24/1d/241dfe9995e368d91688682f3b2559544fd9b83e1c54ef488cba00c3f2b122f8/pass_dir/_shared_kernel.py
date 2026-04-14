import torch
import triton
import triton.language as tl


# ── Shared Triton kernel (used by ALL pass variants via routing) ──────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _shared_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # div → multiply by reciprocal (faster)
    x = x * 0.08838834764831843   # 1.0 / 11.313708498984761
    # relu
    x = tl.maximum(x, 0.0)
    # square
    x = x * x
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def shared_fused_div_relu_square(in_0):
    """Shared wrapper used by all pass variants."""
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _shared_div_relu_square_kernel[grid](in_0, out, n_elements)
    return out