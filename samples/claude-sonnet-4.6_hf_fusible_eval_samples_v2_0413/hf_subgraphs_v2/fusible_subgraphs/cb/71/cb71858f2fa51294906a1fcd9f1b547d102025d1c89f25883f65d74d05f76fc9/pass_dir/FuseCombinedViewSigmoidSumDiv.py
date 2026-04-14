import torch
import triton
import triton.language as tl


def pattern(conv2d_out, in_3):
    # Path A: view(1,2,8,8) then sigmoid  -- conv2d output is [1,128,1,1], 128 elements
    t = conv2d_out.view(1, 2, 8, 8)
    tmp_4 = t.sigmoid()
    # Path B: sum over dim=3, then divide  -- in_3 is [1,2,8,8], 128 elements
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return (tmp_6, tmp_4)


def replacement_args(conv2d_out, in_3):
    return (conv2d_out, in_3)


@triton.jit
def combined_sigmoid_sumdiv_kernel(
    sigmoid_in_ptr,   # conv2d output [1,128,1,1], 128 elements contiguous
    sigmoid_out_ptr,  # sigmoid result [1,2,8,8],  128 elements contiguous
    sumdiv_in_ptr,    # in_3           [1,2,8,8],  128 elements contiguous
    sumdiv_out_ptr,   # div result     [1,2,8,8],  128 elements contiguous
    BLOCK: tl.constexpr,   # 8 — one row of in_3's dim-3
):
    """
    Each program handles one block of BLOCK elements.
    16 programs × 8 elements = 128 elements total.
    Path A: element-wise sigmoid (view is implicit — same linear memory order).
    Path B: sum over BLOCK elements, then divide (row-wise L1 normalisation).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # absolute element offsets

    # ---- Path A: sigmoid ------------------------------------------------
    y = tl.load(sigmoid_in_ptr + offs)
    y_f32 = y.to(tl.float32)
    sig = tl.sigmoid(y_f32).to(y.dtype)
    tl.store(sigmoid_out_ptr + offs, sig)

    # ---- Path B: row-sum then divide ------------------------------------
    x = tl.load(sumdiv_in_ptr + offs)
    x_f32 = x.to(tl.float32)
    row_sum = tl.sum(x_f32, axis=0)            # scalar sum of BLOCK elements
    normed = (x_f32 / row_sum).to(x.dtype)
    tl.store(sumdiv_out_ptr + offs, normed)


@torch.fx.wrap
def fuse_combined(conv2d_out, in_3):
    """
    Replaces:  view + sigmoid  +  sum(dim=3) + div
    With a single Triton kernel launch (16 programs × 8 elements = 128 elements).
    conv2d_out : [1, 128, 1, 1]
    in_3       : [1,   2, 8, 8]
    Returns (tmp_6, tmp_4)  matching the original model return order.
    """
    out_sigmoid = torch.empty((1, 2, 8, 8), dtype=conv2d_out.dtype, device=conv2d_out.device)
    out_norm    = torch.empty_like(in_3)

    BLOCK   = 8
    N_rows  = 16      # 1 * 2 * 8 = 16 rows of 8 elements each
    grid    = (N_rows,)

    combined_sigmoid_sumdiv_kernel[grid](
        conv2d_out, out_sigmoid,
        in_3,       out_norm,
        BLOCK=BLOCK,
    )
    # model returns (tmp_6, tmp_4) = (sum-div result, sigmoid result)
    return (out_norm, out_sigmoid)


def replacement_func():
    return fuse_combined