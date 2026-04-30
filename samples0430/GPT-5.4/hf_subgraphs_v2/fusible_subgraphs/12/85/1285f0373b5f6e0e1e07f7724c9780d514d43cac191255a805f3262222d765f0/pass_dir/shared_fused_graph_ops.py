import torch
import triton
import triton.language as tl
from torch.utils._mode_utils import no_dispatch


def _unwrap_tensor(x):
    if isinstance(x, torch.Tensor) and type(x) is not torch.Tensor:
        with no_dispatch():
            return x.as_subclass(torch.Tensor)
    return x


@triton.jit
def masked_gather_2row_kernel(
    in_ptr,
    idx_ptr,
    out_ptr,
    m,
    in_stride0,
    in_stride1,
    out_stride0,
    out_stride1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < m

    idx = tl.load(idx_ptr + offs, mask=mask, other=0)

    v0 = tl.load(in_ptr + 0 * in_stride0 + idx * in_stride1, mask=mask, other=0)
    v1 = tl.load(in_ptr + 1 * in_stride0 + idx * in_stride1, mask=mask, other=0)

    tl.store(out_ptr + 0 * out_stride0 + offs * out_stride1, v0, mask=mask)
    tl.store(out_ptr + 1 * out_stride0 + offs * out_stride1, v1, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["total"],
)
@triton.jit
def cat_and_fill_2row_kernel(
    left_ptr,
    right_ptr,
    out_cat_ptr,
    out_ones_ptr,
    m,
    total,
    left_stride0,
    left_stride1,
    right_stride0,
    right_stride1,
    out_cat_stride0,
    out_cat_stride1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    left_sel = offs < m
    right_offs = tl.maximum(offs - m, 0)

    left_mask = mask & left_sel
    right_mask = mask & (~left_sel)

    l0 = tl.load(left_ptr + offs * left_stride1, mask=left_mask, other=0)
    l1 = tl.load(left_ptr + left_stride0 + offs * left_stride1, mask=left_mask, other=0)

    r0 = tl.load(right_ptr + right_offs * right_stride1, mask=right_mask, other=0)
    r1 = tl.load(right_ptr + right_stride0 + right_offs * right_stride1, mask=right_mask, other=0)

    o0 = tl.where(left_sel, l0, r0)
    o1 = tl.where(left_sel, l1, r1)

    tl.store(out_cat_ptr + offs * out_cat_stride1, o0, mask=mask)
    tl.store(out_cat_ptr + out_cat_stride0 + offs * out_cat_stride1, o1, mask=mask)
    tl.store(out_ones_ptr + offs, 1.0, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["total"],
)
@triton.jit
def fused_mask_cat_ones_2row_kernel(
    in0_ptr,
    idx_ptr,
    in1_ptr,
    out_cat_ptr,
    out_ones_ptr,
    m,
    total,
    in0_stride0,
    in0_stride1,
    in1_stride0,
    in1_stride1,
    out_cat_stride0,
    out_cat_stride1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    left_sel = offs < m
    right_offs = tl.maximum(offs - m, 0)

    gather_mask = mask & left_sel
    right_mask = mask & (~left_sel)

    idx = tl.load(idx_ptr + offs, mask=gather_mask, other=0)

    g0 = tl.load(in0_ptr + idx * in0_stride1, mask=gather_mask, other=0)
    g1 = tl.load(in0_ptr + in0_stride0 + idx * in0_stride1, mask=gather_mask, other=0)

    r0 = tl.load(in1_ptr + right_offs * in1_stride1, mask=right_mask, other=0)
    r1 = tl.load(in1_ptr + in1_stride0 + right_offs * in1_stride1, mask=right_mask, other=0)

    o0 = tl.where(left_sel, g0, r0)
    o1 = tl.where(left_sel, g1, r1)

    tl.store(out_cat_ptr + offs * out_cat_stride1, o0, mask=mask)
    tl.store(out_cat_ptr + out_cat_stride0 + offs * out_cat_stride1, o1, mask=mask)
    tl.store(out_ones_ptr + offs, 1.0, mask=mask)


@torch.fx.wrap
def graph_fused_dispatch(*args):
    route = args[-1]

    if route == "full":
        in_0, in_1, in_2, _ = args
        raw_in_0 = _unwrap_tensor(in_0)
        raw_in_1 = _unwrap_tensor(in_1)
        raw_in_2 = _unwrap_tensor(in_2)

        with no_dispatch():
            idx = torch.nonzero(raw_in_2, as_tuple=False).flatten()
            m = int(idx.numel())
            total = int(raw_in_1.size(1)) + m

        out_cat = torch.empty((2, total), dtype=raw_in_1.dtype, device=raw_in_1.device)
        out_ones = torch.empty((total,), dtype=torch.float32, device=raw_in_1.device)

        if total > 0:
            grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
            fused_mask_cat_ones_2row_kernel[grid](
                raw_in_0,
                idx,
                raw_in_1,
                out_cat,
                out_ones,
                m,
                total,
                raw_in_0.stride(0),
                raw_in_0.stride(1),
                raw_in_1.stride(0),
                raw_in_1.stride(1),
                out_cat.stride(0),
                out_cat.stride(1),
            )
        return out_cat, out_ones

    if route == "gather_sym_size":
        in_0, in_2, _ = args
        raw_in_0 = _unwrap_tensor(in_0)
        raw_in_2 = _unwrap_tensor(in_2)

        with no_dispatch():
            idx = torch.nonzero(raw_in_2, as_tuple=False).flatten()
            m = int(idx.numel())

        out = torch.empty((2, m), dtype=raw_in_0.dtype, device=raw_in_0.device)
        if m > 0:
            grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE"]),)
            masked_gather_2row_kernel[grid](
                raw_in_0,
                idx,
                out,
                m,
                raw_in_0.stride(0),
                raw_in_0.stride(1),
                out.stride(0),
                out.stride(1),
            )
        return out, m

    if route == "cat_ones":
        tmp_1, in_1, tmp_10, _ = args
        raw_tmp_1 = _unwrap_tensor(tmp_1)
        raw_in_1 = _unwrap_tensor(in_1)
        total = int(tmp_10)
        m = int(raw_tmp_1.size(1))

        out_cat = torch.empty((2, total), dtype=raw_in_1.dtype, device=raw_in_1.device)
        out_ones = torch.empty((total,), dtype=torch.float32, device=raw_in_1.device)

        if total > 0:
            grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
            cat_and_fill_2row_kernel[grid](
                raw_tmp_1,
                raw_in_1,
                out_cat,
                out_ones,
                m,
                total,
                raw_tmp_1.stride(0),
                raw_tmp_1.stride(1),
                raw_in_1.stride(0),
                raw_in_1.stride(1),
                out_cat.stride(0),
                out_cat.stride(1),
            )
        return out_cat, out_ones

    raise RuntimeError(f"Unknown route: {route}")