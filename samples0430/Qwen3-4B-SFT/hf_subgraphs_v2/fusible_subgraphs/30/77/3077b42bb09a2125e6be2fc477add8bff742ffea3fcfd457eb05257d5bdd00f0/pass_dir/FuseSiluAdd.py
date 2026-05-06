import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: silu(in_1, inplace=True) + in_0
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0 + in_0
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused SiLU(x) + residual addition
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["n_elements"],
)
@triton.jit
def _silu_add_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    x1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x)  — compute in fp32 for numerical stability
    x1_f32 = x1.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x1_f32))
    silu_x1 = x1_f32 * sig

    # Fused residual add, cast back to original dtype
    out = silu_x1.to(x0.dtype) + x0

    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_silu_add(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _silu_add_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        n_elements=N,
    )
    return out


# ---------------------------------------------------------------------------
# Replacement entry-point
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_silu_add