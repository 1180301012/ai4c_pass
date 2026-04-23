import torch
import triton
import triton.language as tl
from torch import inf
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


# Pattern matching function
# Matches the exact normalization/index/multiply branch shared by all target graphs.
def pattern(in_2, in_3, in_4, in_5):
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(inf)
    tmp_2 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8


# Argument extraction function
def replacement_args(in_2, in_3, in_4, in_5):
    return (in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["n_elements"],
)
@triton.jit
def _deg_norm_inplace_kernel(
    deg_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(deg_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / tl.sqrt(x)
    y = tl.where(y == float("inf"), 0.0, y)
    tl.store(deg_ptr + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=["n_edges"],
)
@triton.jit
def _edge_norm_mul_kernel(
    col_ptr,
    deg_ptr,
    edge_weight_ptr,
    row_ptr,
    out_ptr,
    n_edges,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_edges

    row_idx = tl.load(row_ptr + offs, mask=mask, other=0).to(tl.int32)
    col_idx = tl.load(col_ptr + offs, mask=mask, other=0).to(tl.int32)

    row_deg = tl.load(deg_ptr + row_idx, mask=mask, other=0.0).to(tl.float32)
    col_deg = tl.load(deg_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    edge_w = tl.load(edge_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = row_deg * edge_w * col_deg
    tl.store(out_ptr + offs, out, mask=mask)


# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_graph_degree_norm_index_mul(col, deg, edge_weight, row):
    col_t = unwrap_tensor(col)
    deg_t = unwrap_tensor(deg)
    edge_weight_t = unwrap_tensor(edge_weight)
    row_t = unwrap_tensor(row)

    n_nodes = deg_t.numel()
    if n_nodes > 0:
        grid = lambda meta: (triton.cdiv(n_nodes, meta["BLOCK_SIZE"]),)
        _deg_norm_inplace_kernel[grid](
            deg_t,
            n_nodes,
        )

    out = unwrap_tensor(torch.empty_like(edge_weight))
    n_edges = edge_weight_t.numel()
    if n_edges > 0:
        grid = lambda meta: (triton.cdiv(n_edges, meta["BLOCK_SIZE"]),)
        _edge_norm_mul_kernel[grid](
            col_t,
            deg_t,
            edge_weight_t,
            row_t,
            out,
            n_edges,
        )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_graph_degree_norm_index_mul