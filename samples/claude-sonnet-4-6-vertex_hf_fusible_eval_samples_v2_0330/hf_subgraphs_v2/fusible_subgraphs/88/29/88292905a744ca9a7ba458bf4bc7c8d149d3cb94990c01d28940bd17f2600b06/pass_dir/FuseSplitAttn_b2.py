import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=4),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['HW', 'C'],
)
@triton.jit
def _fused_split_attn_b2_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, C, HW,
    stride_in0_b, stride_in0_k, stride_in0_c,
    stride_in1_b, stride_in1_k,
    stride_out_b, stride_out_c,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel for Split-Attention weighted sum:
      out[b, c, hw] = softmax(in1[b, :, 0, c])[0] * in0[b, 0, c, hw]
                    + softmax(in1[b, :, 0, c])[1] * in0[b, 1, c, hw]
    """
    b        = tl.program_id(0)
    c        = tl.program_id(1)
    hw_block = tl.program_id(2)

    hw_start = hw_block * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask  = hw_offs < HW

    # ---- Softmax over 2 elements (in fp32 for numerical stability) ----
    w0 = tl.load(in1_ptr + b * stride_in1_b + c).to(tl.float32)
    w1 = tl.load(in1_ptr + b * stride_in1_b + stride_in1_k + c).to(tl.float32)

    max_w   = tl.maximum(w0, w1)
    e0      = tl.exp(w0 - max_w)
    e1      = tl.exp(w1 - max_w)
    inv_sum = 1.0 / (e0 + e1)
    w0      = e0 * inv_sum
    w1      = e1 * inv_sum

    # ---- Load in0 for k=0 and k=1 ----
    base0 = in0_ptr + b * stride_in0_b + c * stride_in0_c
    base1 = in0_ptr + b * stride_in0_b + stride_in0_k + c * stride_in0_c

    v0 = tl.load(base0 + hw_offs, mask=hw_mask, other=0.0).to(tl.float32)
    v1 = tl.load(base1 + hw_offs, mask=hw_mask, other=0.0).to(tl.float32)

    # ---- Weighted sum ----
    result = w0 * v0 + w1 * v1

    # ---- Store (Triton implicitly casts fp32 to output element dtype) ----
    tl.store(out_ptr + b * stride_out_b + c * stride_out_c + hw_offs,
             result, mask=hw_mask)


@torch.fx.wrap
def fused_split_attn_b2(in_0, in_1):
    """
    Replacement for:
      softmax(in_1, dim=1) -> reshape -> view -> view -> * in_0 -> sum(dim=1) -> contiguous()

    in_0 : [B, 2, C, H, W]
    in_1 : [B, 2, 1, C]
    out  : [B, C, H, W]
    """
    B  = in_0.shape[0]
    C  = in_0.shape[2]
    H  = in_0.shape[3]
    W  = in_0.shape[4]
    HW = H * W

    out = torch.empty(B, C, H, W, dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (B, C, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_split_attn_b2_kernel[grid](
        in_0, in_1, out,
        B, C, HW,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1),
        out.stride(0),  out.stride(1),
    )

    return out


# ---------------------------------------------------------------------------
# Pattern – matches batch-size == 2 graphs with CONCRETE reshape/view sizes.
# For C=128, K=2: 2*1*128 = 256 total weight elements per batch sample.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(2, 256)
    tmp_2 = tmp_1.view(2, 256, 1, 1)
    tmp_3 = tmp_2.view(2, 2, 128, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_split_attn_b2