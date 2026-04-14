import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=['BHW', 'C'],
)
@triton.jit
def _fused_einsum_cat_softmax_kernel(
    in0_ptr, in1_ptr, in2_ptr,
    out3_ptr,
    BHW, H, W, C, J,
    s_in0_b, s_in0_h, s_in0_w,
    s_in1_b, s_in1_c, s_in1_h,
    s_in2_b, s_in2_c, s_in2_h,
    s_out3_b, s_out3_h, s_out3_w,
    BLOCK_J: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Each program handles one (b, h, w) output row
    pid = tl.program_id(0)
    b = pid // (H * W)
    hw = pid % (H * W)
    h = hw // W
    w = hw % W

    j_offs = tl.arange(0, BLOCK_J)  # [0, 1, ..., 63]

    # Compute einsum: acc[j] = sum_c in2[b,c,h,w] * in1[b,c,h,j]
    acc = tl.zeros([BLOCK_J], dtype=tl.float32)
    in1_base = b * s_in1_b + h * s_in1_h
    in2_base = b * s_in2_b + h * s_in2_h + w

    for c in range(C):
        # Scalar load: in2[b, c, h, w]
        a = tl.load(in2_ptr + in2_base + c * s_in2_c).to(tl.float32)
        # Vector load: in1[b, c, h, j] for all j (contiguous in j)
        bv = tl.load(in1_ptr + in1_base + c * s_in1_c + j_offs).to(tl.float32)
        acc += a * bv

    # Load in0[b, h, w, j] (first 64 elements before concat)
    in0_base = b * s_in0_b + h * s_in0_h + w * s_in0_w
    in0_vals = tl.load(in0_ptr + in0_base + j_offs).to(tl.float32)

    # Softmax over concatenated [in0_vals (64), acc (64)] = 128 elements
    # Numerically stable: subtract max before exp
    mx = tl.maximum(tl.max(in0_vals, axis=0), tl.max(acc, axis=0))
    e0 = tl.exp(in0_vals - mx)
    e1 = tl.exp(acc - mx)
    s_sum = tl.sum(e0, axis=0) + tl.sum(e1, axis=0)

    sm0 = (e0 / s_sum).to(DTYPE)   # softmax of in0 half  → out3[..., 0:64]
    sm1 = (e1 / s_sum).to(DTYPE)   # softmax of einsum half → out3[..., 64:128]

    # Store to output: out3[b, h, w, 0:64] and out3[b, h, w, 64:128]
    out3_base = b * s_out3_b + h * s_out3_h + w * s_out3_w
    tl.store(out3_ptr + out3_base + j_offs, sm0)
    tl.store(out3_ptr + out3_base + 64 + j_offs, sm1)


@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    B, C, H, W = in_2.shape
    J = in_1.shape[3]   # in_1: [B, C, H, J]
    BHW = B * H * W

    # Output: [B, H, W, 128] — first 64 is softmax(in_0 half), next 64 is softmax(einsum half)
    out3 = torch.empty((B, H, W, 128), dtype=in_0.dtype, device=in_0.device)

    # Select Triton output dtype constexpr
    if in_0.dtype == torch.float16:
        DTYPE = tl.float16
    elif in_0.dtype == torch.bfloat16:
        DTYPE = tl.bfloat16
    else:
        DTYPE = tl.float32

    _fused_einsum_cat_softmax_kernel[(BHW,)](
        in_0, in_1, in_2, out3,
        BHW, H, W, C, J,
        # in0 strides (dim 0,1,2 — dim 3 is stride-1 and handled via j_offs)
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        # in1 strides (dim 0,1,2)
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        # in2 strides (dim 0,1,2)
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        # out3 strides (dim 0,1,2)
        out3.stride(0), out3.stride(1), out3.stride(2),
        BLOCK_J=64,
        DTYPE=DTYPE,
    )

    # tmp_4 = tmp_3[..., :64] — non-contiguous view of out3 (matches baseline slice semantics)
    out4 = out3[..., :64]
    return (out3, out4)


def replacement_func():
    return fused_einsum_cat_softmax