"""
Shared dispatch module for embedding+permute+expand fusion passes.
All pass files import `embed_dispatch` from here so they reference the
EXACT same function object — satisfying the replacement_func_limit=1 constraint.

Patterns INCLUDE .to(device) so the full chain is one graph node,
minimising Python dispatch overhead. in_1 arrives as a CPU tensor.
All constants are hardcoded to minimise per-call Python work.
"""
import torch
import triton
import triton.language as tl


# ── Kernel for batch=1 cases: 2-D grid (seq_len, emb_dim) ───────────────────
@triton.jit
def _kernel_embed_rowwise(
    weight_ptr, indices_ptr, output_ptr,
    EMB_DIM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    SEQ_SQ:  tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)   # row  in [0, SEQ_LEN)
    d = tl.program_id(1)   # dim  in [0, EMB_DIM)

    j_offs = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offs < SEQ_LEN

    flat_ij = i * SEQ_LEN + j_offs
    idx  = tl.load(indices_ptr + flat_ij, mask=j_mask, other=0)
    vals = tl.load(weight_ptr + idx * EMB_DIM + d, mask=j_mask, other=0.0)
    tl.store(output_ptr + d * SEQ_SQ + i * SEQ_LEN + j_offs, vals, mask=j_mask)


# ── Kernel for batch=2: 2-D grid (seq_len, emb_dim*2) ───────────────────────
@triton.jit
def _kernel_embed_batch2(
    weight_ptr, indices_ptr, output_ptr,
    EMB_DIM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    SEQ_SQ:  tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i  = tl.program_id(0)
    bd = tl.program_id(1)
    b  = bd // EMB_DIM
    d  = bd % EMB_DIM

    j_offs = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offs < SEQ_LEN

    flat_ij = i * SEQ_LEN + j_offs
    idx  = tl.load(indices_ptr + flat_ij, mask=j_mask, other=0)
    vals = tl.load(weight_ptr + idx * EMB_DIM + d, mask=j_mask, other=0.0)
    tl.store(output_ptr + b * EMB_DIM * SEQ_SQ + d * SEQ_SQ + i * SEQ_LEN + j_offs,
             vals, mask=j_mask)


# ── Shared dispatch wrapper ───────────────────────────────────────────────────
@torch.fx.wrap
def embed_dispatch(in_0, in_1, route):
    """
    in_0 : embedding weight  (CUDA for fp16/bf16 models; CPU for all-mpnet-base-v2)
    in_1 : indices            (CPU — the .to(device) op is included in the pattern)
    route : "1_45" | "1_11" | "2_7"
    All shape / device constants are hardcoded to minimise Python overhead.
    """
    if route == "1_45":
        # weight already CUDA; emb_dim=4, seq=45
        indices = torch.as_tensor(in_1, device='cuda')
        out = torch.empty((1, 4, 45, 45), dtype=in_0.dtype, device='cuda')
        _kernel_embed_rowwise[(45, 4)](
            in_0, indices, out,
            EMB_DIM=4, SEQ_LEN=45, SEQ_SQ=2025, BLOCK_SIZE=64)
        return out

    elif route == "1_11":
        # weight already CUDA; emb_dim=12, seq=11
        indices = torch.as_tensor(in_1, device='cuda')
        out = torch.empty((1, 12, 11, 11), dtype=in_0.dtype, device='cuda')
        _kernel_embed_rowwise[(11, 12)](
            in_0, indices, out,
            EMB_DIM=12, SEQ_LEN=11, SEQ_SQ=121, BLOCK_SIZE=16)
        return out

    elif route == "2_7":
        # weight may be CPU (all-mpnet-base-v2); emb_dim=12, seq=7, batch=2
        weight  = in_0 if in_0.is_cuda else torch.as_tensor(in_0, device='cuda')
        indices = torch.as_tensor(in_1, device='cuda')
        out = torch.empty((2, 12, 7, 7), dtype=weight.dtype, device='cuda')
        _kernel_embed_batch2[(7, 24)](
            weight, indices, out,
            EMB_DIM=12, SEQ_LEN=7, SEQ_SQ=49, BLOCK_SIZE=8)
        return out