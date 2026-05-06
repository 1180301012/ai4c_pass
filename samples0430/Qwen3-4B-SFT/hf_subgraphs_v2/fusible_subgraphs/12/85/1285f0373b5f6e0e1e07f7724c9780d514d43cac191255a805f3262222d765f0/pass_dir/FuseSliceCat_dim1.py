import torch
import triton
import triton.language as tl
from torch import device


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 32}),
        triton.Config({"BLOCK": 64}),
        triton.Config({"BLOCK": 128}),
        triton.Config({"BLOCK": 256}),
    ],
    key=["C0", "C1"],
)
@triton.jit
def fused_slice_cat_kernel(
    in0_ptr,          # [ROWS, C0] int64  on CUDA
    in2_ptr,          # [C0]       bool   on CUDA  (the mask)
    in1_ptr,          # [ROWS, C1] int64  on CUDA
    out_ptr,          # [ROWS, C0+C1] int64 on CUDA
    C0, C1,
    D0: tl.constexpr,  # stride of in_0  (= C0)
    D1: tl.constexpr,  # stride of in_1  (= C1)
    BLOCK: tl.constexpr,
):
    """
    Each program handles one row.
    out[row, 0:C0] = in0[row, :]  if in2[0:C0], else 0
    out[row, C0:C0+C1] = in1[row, :]
    """
    pid  = tl.program_id(0)   # one program per row
    offs = tl.arange(0, BLOCK)

    # ---- Part 1: copy in_0[:, in_2] → out[pid, 0:C0] ----
    mask_in0 = offs < C0
    in0_cols  = pid * D0 + offs
    in2_vals  = tl.load(in2_ptr + offs, mask=mask_in0, other=0)
    val_in0   = tl.load(in0_ptr + in0_cols, mask=mask_in0 & in2_vals, other=0)

    out_row_base = pid * (C0 + C1)
    tl.store(out_ptr + out_row_base + offs, val_in0, mask=mask_in0)

    # ---- Part 2: copy in_1 → out[pid, C0:C0+C1] ----
    mask_in1 = offs < C1
    in1_cols  = pid * D1 + offs
    val_in1   = tl.load(in1_ptr + in1_cols, mask=mask_in1, other=0)
    tl.store(out_ptr + out_row_base + C0 + offs, val_in1, mask=mask_in1)




@torch.fx.wrap
def fused_slice_cat(in_0, in_1, in_2):
    """
    Fused replacement for:
        tmp_1 = in_0[slice(None, None, None), in_2]
        tmp_9 = torch.cat([tmp_1, in_1], dim=1)

    in_0 : [R, C0] bool/int  – may live on CPU
    in_1 : [R, C1] int64     – CUDA
    in_2 : [C0]    bool      – may live on CPU
    Returns out : [R, C0+C1] int64  on CUDA
    """
    R     = in_1.shape[0]
    C0    = in_0.shape[1]
    C1    = in_1.shape[1]

    # Move the (possibly CPU-resident) tensors to CUDA
    in_0_cuda = torch.as_tensor(in_0, device=in_1.device)
    in_2_cuda = torch.as_tensor(in_2, device=in_1.device)

    out   = torch.empty((R, C0 + C1), dtype=torch.int64, device=in_1.device)
    grid  = lambda meta: (R,)

    fused_slice_cat_kernel[grid](
        in_0_cuda, in_2_cuda, in_1, out,
        C0, C1,
        D0=C0, D1=C1,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern – matches the full slice → cat subgraph in BOTH RECT_L and GAE
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_slice_cat