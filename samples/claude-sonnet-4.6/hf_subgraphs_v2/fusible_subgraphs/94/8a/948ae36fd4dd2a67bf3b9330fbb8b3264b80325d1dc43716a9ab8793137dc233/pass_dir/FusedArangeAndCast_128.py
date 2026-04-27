"""
Fused pass: torch.arange(0,128,device=dev) + in_0.to(device=dev,dtype=bool)
→ single Triton kernel launch.

Fix for bool pointer: use torch.int8 output internally, then view as bool (zero-copy).
This avoids Triton's tl.int1 vs PyTorch's byte-per-element bool mismatch.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}),
        triton.Config({"BLOCK": 256}),
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["ar_n", "cast_n"],
)
@triton.jit
def _fused128_kernel(
    in_ptr,       # int64* source for cast
    ar_ptr,       # int64* arange output
    cast_ptr,     # int8*  cast output (stores 0/1, NOT bool ptr)
    ar_n,
    cast_n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    ar_blks = (ar_n + BLOCK - 1) // BLOCK

    if pid < ar_blks:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < ar_n
        tl.store(ar_ptr + offs, offs.to(tl.int64), mask=mask)
    else:
        adj = pid - ar_blks
        offs = adj * BLOCK + tl.arange(0, BLOCK)
        mask = offs < cast_n
        x = tl.load(in_ptr + offs, mask=mask, other=0)
        out = (x != 0).to(tl.int8)
        tl.store(cast_ptr + offs, out, mask=mask)


def pattern(in_0, dev):
    tmp_1 = torch.arange(0, 128, device=dev)
    tmp_2 = in_0.to(device=dev, dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(in_0, dev):
    return (in_0, dev)


@torch.fx.wrap
def _fused_arange_cast_128(in_0, dev):
    N = 128
    M = in_0.numel()
    ar_out = torch.empty(N, dtype=torch.int64, device=dev)
    # Use int8 to avoid Triton bool-pointer issues; reinterpret as bool via view (zero-copy)
    cast_i8 = torch.empty(M, dtype=torch.int8, device=in_0.device)

    grid = lambda meta: (
        (N + meta["BLOCK"] - 1) // meta["BLOCK"]
        + (M + meta["BLOCK"] - 1) // meta["BLOCK"],
    )
    _fused128_kernel[grid](in_0, ar_out, cast_i8, N, M)

    # Zero-copy reinterpret int8→bool then reshape to original layout
    cast_out = cast_i8.view(in_0.shape).view(torch.bool)
    return ar_out, cast_out


def replacement_func():
    return _fused_arange_cast_128