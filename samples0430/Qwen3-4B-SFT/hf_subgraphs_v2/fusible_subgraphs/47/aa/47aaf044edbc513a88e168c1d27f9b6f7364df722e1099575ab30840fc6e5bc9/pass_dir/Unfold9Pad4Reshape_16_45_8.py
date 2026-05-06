import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------
# Pass: Unfold 9x1 with pad [4,0], then reshape [16,45] -> [-1,8,9]
# Matches: ConvBertForSequenceClassification graphs
# in_0: [1, 16, 45]  or  [1, 16, 45] in any fp dtype
# out:  [144, 9]
# -----------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0,)


# -----------------------------------------------------------------------
# Triton kernel:
#   Output flat index offs -> decompose into (win, kpos) in [H*9, 9]
#   Input address = [i0, i1] in the [H, L] matrix
#   stride_row = in_0.stride(1)  (row stride of the 2D input view)
#   stride_col = in_0.stride(2)  (col stride)
# 
# The batch dim (=1 here) is ignored; data ptr starts at element [0,0,0].
# -----------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
        triton.Config({'BLOCK': 512}),
    ],
    key=['total'],
)
@triton.jit
def _unf9r4r45_kernel(
    in_ptr,
    out_ptr,
    L,
    H_flat,
    W,
    stride_row,
    stride_col,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Decompose linear output index into (win_flat, kpos)
    win_flat = offs // W      # 0 .. H*9-1
    kpos     = offs  % W      # 0 .. 8

    h      = win_flat  // 9
    r_win  = win_flat  %  9  # which window: 0..8

    # Source seq/height position
    i_range = r_win - 4      # element index within L

    # Load positions kpos .. kpos+8 from the input
    # mask covers out-of-range (i_range) and (kpos+8 out of L)
    i_base  = i_range * stride_row + h * stride_row   # offset to row h
    addr    = i_base + (tl.arange(0, 9) + kpos) * stride_col
    mask2   = (i_range >= 0) & (i_range < L)
    vals    = tl.load(in_ptr + addr, mask=mask2, other=0.0)

    tl.store(out_ptr + offs, vals, mask=mask)


@torch.fx.wrap
def _unf9r4r45_wrapper(in_0):
    H         = 16
    L         = 45
    W         = 9
    H_flat    = H * W   # 144
    total     = H_flat * W  # 1296

    out   = torch.empty((H_flat, W), dtype=in_0.dtype, device=in_0.device)
    # stride_row / stride_col for the 2D [H, L] view of the tensor
    stride_row = in_0.stride(1) if in_0.stride(1) != 0 else L
    stride_col = in_0.stride(2) if in_0.stride(2) != 0 else 1

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK']),)
    _unf9r4r45_kernel[grid](
        in_0, out,
        L, H_flat, W,
        stride_row, stride_col,
        total,
    )
    return out


def replacement_func():
    return _unf9r4r45_wrapper