import torch
import triton
import triton.language as tl


@triton.jit
def _add_softmax_kernel_625_625(
    in0_ptr, in1_ptr, out_ptr,
    H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + softmax kernel for H=625, W=625 (float32)."""
    row_id = tl.program_id(0)
    row = row_id // H
    i = row_id % H

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < W

    # Load in_0 as [1,1,H,W]: flat index for row i, col k is  i*W + k
    x0 = tl.load(in0_ptr + i * W + cols, mask=mask, other=0.0)

    # Load in_1 as [1,8,H,W]: flat index for row (r*H+i), col k is (row*H+i)*W + k
    x1 = tl.load(in1_ptr + row_id * W + cols, mask=mask, other=0.0)

    # Add in float32 for numerical stability
    x = x0.to(tl.float32) + x1.to(tl.float32)

    # Online softmax: subtract max, exp, sum, normalize
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    x_sum = tl.sum(x, axis=0)
    x = x / x_sum

    # Store result (cast back to original dtype — float32 for this pass)
    tl.store(out_ptr + row_id * W + cols, x, mask=mask)


@torch.fx.wrap
def fused_add_softmax_625_625(in_0, in_1):
    """
    Fused kernel replacing: add + view(8,625,625) + softmax + view(1,8,625,625)
                              + view(8,625,625) + dropout(p=0)
    Returns (output, softmax_result) matching model's (tmp_5, tmp_3).
    """
    B, H, W = 8, 625, 625
    out = torch.empty((1, 8, H, W), dtype=in_1.dtype, device=in_1.device)

    # Grid: one program per row (B*H = 5000 rows)
    _add_softmax_kernel_625_625[(B * H,)](
        in_0, in_1, out,
        H, W,
        BLOCK_SIZE=1024,
    )

    # Both tmp_3 and tmp_5 are identical (dropout p=0.0 training=False is identity)
    tmp3 = out.view(1, B, H, W)
    tmp5 = out.view(B, H, W)
    return tmp5, tmp3


# ---- Pattern to match ----

def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 625, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return tmp_5, tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_add_softmax_625_625