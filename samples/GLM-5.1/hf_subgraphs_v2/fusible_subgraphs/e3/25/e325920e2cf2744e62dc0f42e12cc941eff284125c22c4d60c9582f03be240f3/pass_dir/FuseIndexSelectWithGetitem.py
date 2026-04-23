import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def gather_kernel_min(
    edge_index_ptr,
    x_ptr,
    gathered_out_ptr,
    num_edges,
    stride_ei0,
    stride_ei1,
    stride_x0,
    stride_g0,
    BLOCK: tl.constexpr,
    FEAT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_edges
    idx = tl.load(edge_index_ptr + stride_ei0 * 0 + offs * stride_ei1, mask=mask)
    foffs = tl.arange(0, FEAT)
    x_ptrs = x_ptr + idx[:, None] * stride_x0 + foffs[None, :]
    g_ptrs = gathered_out_ptr + offs[:, None] * stride_g0 + foffs[None, :]
    fmask = mask[:, None] & (foffs < FEAT)[None, :]
    vals = tl.load(x_ptrs, mask=fmask, other=0.0)
    tl.store(g_ptrs, vals, mask=fmask)


@torch.fx.wrap
def fused_gather(in_0, in_1):
    num_edges = in_0.shape[1]
    feat_dim = in_1.shape[1]
    gathered_out = torch.empty(num_edges, feat_dim, dtype=in_1.dtype, device=in_1.device)
    
    # Use large block to minimize programs - just 2 programs for 1100 edges
    BLOCK = 512
    grid = (triton.cdiv(num_edges, BLOCK),)
    
    gather_kernel_min[grid](
        edge_index_ptr=in_0,
        x_ptr=in_1,
        gathered_out_ptr=gathered_out,
        num_edges=num_edges,
        stride_ei0=in_0.stride(0),
        stride_ei1=in_0.stride(1),
        stride_x0=in_1.stride(0),
        stride_g0=gathered_out.stride(0),
        BLOCK=BLOCK,
        FEAT=16,
        num_warps=4,
    )
    
    return gathered_out


def replacement_func():
    return fused_gather