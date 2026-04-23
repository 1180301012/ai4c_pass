import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_cat_nearest_interpolate_stack40_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    total_elems,
    N,
    in0_s0,
    in0_s1,
    in0_s2,
    in0_s3,
    in1_s0,
    in1_s1,
    in1_s2,
    in1_s3,
    in2_s0,
    in2_s1,
    in2_s2,
    in2_s3,
    in3_s0,
    in3_s1,
    in3_s2,
    in3_s3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elems

    x = offs
    w = x % 40
    x = x // 40
    h = x % 40
    x = x // 40
    c = x % 512
    x = x // 512
    n = x % N
    s = x // N

    mask0 = mask & (s == 0)
    mask1 = mask & (s == 1)
    mask2 = mask & (s == 2) & (c < 256)
    mask3 = mask & (s == 2) & (c >= 256)

    h1 = h // 2
    w1 = w // 2
    c2 = tl.where(c < 256, c, 0)
    c3 = tl.where(c >= 256, c - 256, 0)

    off0 = n * in0_s0 + c * in0_s1 + h * in0_s2 + w * in0_s3
    off1 = n * in1_s0 + c * in1_s1 + h1 * in1_s2 + w1 * in1_s3
    off2 = n * in2_s0 + c2 * in2_s1 + h * in2_s2 + w * in2_s3
    off3 = n * in3_s0 + c3 * in3_s1 + h * in3_s2 + w * in3_s3

    v0 = tl.load(in0_ptr + off0, mask=mask0, other=0.0).to(tl.float32)
    v1 = tl.load(in1_ptr + off1, mask=mask1, other=0.0).to(tl.float32)
    v2 = tl.load(in2_ptr + off2, mask=mask2, other=0.0).to(tl.float32)
    v3 = tl.load(in3_ptr + off3, mask=mask3, other=0.0).to(tl.float32)

    out_val = v0 + v1 + v2 + v3
    tl.store(out_ptr + offs, out_val, mask=mask)


@torch.fx.wrap
def fused_cat_nearest_interpolate_stack40(in_0, in_1, in_2, in_3):
    n = in_0.shape[0]
    out = torch.empty((3, n, 512, 40, 40), device=in_0.device, dtype=in_0.dtype)
    total_elems = out.numel()

    in0_s0, in0_s1, in0_s2, in0_s3 = in_0.stride()
    in1_s0, in1_s1, in1_s2, in1_s3 = in_1.stride()
    in2_s0, in2_s1, in2_s2, in2_s3 = in_2.stride()
    in3_s0, in3_s1, in3_s2, in3_s3 = in_3.stride()

    grid = lambda META: (triton.cdiv(total_elems, META['BLOCK_SIZE']),)

    fused_cat_nearest_interpolate_stack40_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        out,
        total_elems,
        n,
        in0_s0,
        in0_s1,
        in0_s2,
        in0_s3,
        in1_s0,
        in1_s1,
        in1_s2,
        in1_s3,
        in2_s0,
        in2_s1,
        in2_s2,
        in2_s3,
        in3_s0,
        in3_s1,
        in3_s2,
        in3_s3,
    )
    return out


def replacement_func():
    return fused_cat_nearest_interpolate_stack40