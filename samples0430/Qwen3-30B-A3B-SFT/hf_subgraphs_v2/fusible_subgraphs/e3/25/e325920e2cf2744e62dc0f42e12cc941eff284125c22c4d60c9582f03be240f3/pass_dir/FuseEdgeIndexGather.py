import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches index_select(-2, indices).
# Single-input single-output avoids the pass framework's bug where a single
# placeholder is used in two getitem nodes in the same pattern.
# ---------------------------------------------------------------------------
def pattern(in_1, tmp_1):
    tmp_2 = in_1.index_select(-2, tmp_1)
    return tmp_2


def replacement_args(in_1, tmp_1):
    return (in_1, tmp_1)


# ---------------------------------------------------------------------------
# Triton kernel: simple 1-D gather.
#   Grid = (E,): one program per edge.
#   D = 16 constexpr → Triton vectorises load/store as a single 32-byte txn.
#   num_warps=1 (32 threads, 16 active) — exact fit for D=16.
# ---------------------------------------------------------------------------
@triton.jit
def gather_dim1_kernel(
    src_ptr,            # [N, D]  float16 / bfloat16
    idx_ptr,            # [E]     int64
    out_ptr,            # [E, D]  same dtype as src
    E,                  # number of edges
    D: tl.constexpr,    # feature dim (16)
):
    pid = tl.program_id(0)
    d   = tl.arange(0, D)

    src_row = tl.load(idx_ptr + pid).to(tl.int32)
    vals    = tl.load(src_ptr + src_row * D + d)
    tl.store(out_ptr + pid * D + d, vals)


@torch.fx.wrap
def triton_index_select(in_1, tmp_1):
    E = tmp_1.shape[0]    # 1100
    D = in_1.shape[1]     # 16

    out = torch.empty(E, D, dtype=in_1.dtype, device=in_1.device)

    gather_dim1_kernel[(E,)](
        in_1, tmp_1, out,
        E, D,
        num_warps=1,
        num_stages=1,
    )

    return out


def replacement_func():
    return triton_index_select