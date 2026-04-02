import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU kernel: out = x * sigmoid(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Upcast to float32 for numerically stable sigmoid
    x_f32 = x.to(tl.float32)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x_f32))
    out_f32 = x_f32 * sigmoid_x

    # Cast back to original dtype before storing
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_silu_detach(in_0):
    """
    Computes SiLU(in_0) with a fast Triton kernel.
    Returns (out, out) to cover both the detach output (tmp_3) and the
    raw silu output (tmp_0) that the pattern exposes.
    """
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: (
        (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],
    )
    _silu_kernel[grid](in_0, out, n_elements)

    # detach() is a no-op at inference; both tmp_3 and tmp_0 are the same data
    return (out, out)


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(in_0):
    """
    Matches the subgraph:
        tmp_0 = silu(in_0, inplace=True)
        tmp_3 = tmp_0.detach()
    and returns both observable outputs (tmp_3, tmp_0).
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_3 = tmp_0.detach()
    return (tmp_3, tmp_0)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return triton_silu_detach