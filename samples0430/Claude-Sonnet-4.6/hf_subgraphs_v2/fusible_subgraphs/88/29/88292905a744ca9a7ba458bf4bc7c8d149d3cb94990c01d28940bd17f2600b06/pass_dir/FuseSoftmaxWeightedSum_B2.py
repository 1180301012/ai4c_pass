import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _sws_b2_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, C, HW,
    in0_sB, in0_sK, in0_sC,
    in1_sB, in1_sK,
    out_sB, out_sC,
    USE_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax-weighted sum kernel.
    For each (b, c): weight = softmax(in1[b, :, 0, c])
                     out[b, c, hw] = weight[0]*in0[b,0,c,hw] + weight[1]*in0[b,1,c,hw]
    Grid: (B*C, ceil(HW/BLOCK_SIZE))
    """
    bc_idx = tl.program_id(0)
    hw_pid = tl.program_id(1)

    b = bc_idx // C
    c = bc_idx  % C

    hw_off = hw_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask   = hw_off < HW

    # Load softmax logits in float32 for numerical stability
    w0 = tl.load(in1_ptr + b * in1_sB + c).to(tl.float32)
    w1 = tl.load(in1_ptr + b * in1_sB + in1_sK + c).to(tl.float32)

    # Numerically-stable softmax over 2 values
    max_w   = tl.maximum(w0, w1)
    e0      = tl.exp(w0 - max_w)
    e1      = tl.exp(w1 - max_w)
    inv_sum = 1.0 / (e0 + e1)
    s0 = e0 * inv_sum
    s1 = e1 * inv_sum

    # Load in_0 slices for k=0 and k=1
    base0 = b * in0_sB + c * in0_sC + hw_off
    base1 = b * in0_sB + in0_sK + c * in0_sC + hw_off
    v0 = tl.load(in0_ptr + base0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(in0_ptr + base1, mask=mask, other=0.0).to(tl.float32)

    # Weighted sum
    out_val = s0 * v0 + s1 * v1

    # Cast back to output dtype
    if USE_FP16 == 1:
        out_val = out_val.to(tl.float16)
    elif USE_FP16 == 2:
        out_val = out_val.to(tl.bfloat16)

    tl.store(out_ptr + b * out_sB + c * out_sC + hw_off, out_val, mask=mask)


@torch.fx.wrap
def _dispatch_sws_b2(in_0, in_1):
    B, _, C, H, W = in_0.shape
    HW = H * W

    out = torch.empty(B, C, H, W, dtype=in_0.dtype, device=in_0.device)

    dtype_flag = (1 if in_0.dtype == torch.float16
                  else 2 if in_0.dtype == torch.bfloat16
                  else 0)

    grid = lambda meta: (B * C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    _sws_b2_kernel[grid](
        in_0, in_1, out,
        B, C, HW,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1),
        out.stride(0), out.stride(1),
        dtype_flag,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(2, -1)
    tmp_2 = tmp_1.view(2, -1, 1, 1)
    tmp_3 = tmp_2.view(2, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return _dispatch_sws_b2