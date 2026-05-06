import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    tmp_6 = torch.cat([tmp_5, tmp_4], dim=1)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=8, num_stages=2),
    ],
    key=['HW2'],
)
@triton.jit
def _fused_relu_scale_bias_maxpool_cat(
    in0_ptr,   # [1] bias (scale)
    in1_ptr,   # [1] scale
    in2_ptr,   # [B, C2, H2, W2]
    in3_ptr,   # [B, C2, H3, W3]
    out_ptr,   # [B, C_out, H2, W2]  where C_out = 2*C2
    H2, W2, H3, W3,
    C_out,
    BLOCK_W: tl.constexpr,
):
    # Grid: (ceil(HW2/BLOCK_W), C_out * B)
    pid_w = tl.program_id(0)

    # Each program handles one (b, c) pair with a contiguous block along W
    pid_bc = tl.program_id(1)
    b = pid_bc // C_out
    c = pid_bc % C_out

    HW2 = H2 * W2

    w_start = pid_w * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask = w_offsets < HW2

    # ---- max_pool2d write (cat's first half, c < C2) ----
    if c < C2:
        h = w_offsets // W2   # [BLOCK_W]
        w_in = w_offsets % W2  # [BLOCK_W]

        h0 = h - 1
        h1 = h + 1
        w0 = w_in - 1
        w1 = w_in + 1

        h0_include = (h0 >= 0) & (h0 < H3)
        h1_include = (h1 >= 0) & (h1 < H3)
        w0_include = (w0 >= 0) & (w0 < W3)
        w1_include = (w1 >= 0) & (w1 < W3)

        h0_cl = tl.where(h0_include, h0, 0)
        h1_cl = tl.where(h1_include, h1, 0)
        w0_cl = tl.where(w0_include, w0, 0)
        w1_cl = tl.where(w1_include, w1, 0)

        in3_base = b * C2 * H3 * W3 + c * H3 * W3

        v00 = tl.load(in3_ptr + in3_base + h0_cl * W3 + w0_cl,
                      mask=h0_include & w0_include & mask, other=0.0)
        v01 = tl.load(in3_ptr + in3_base + h0_cl * W3 + w1_cl,
                      mask=h0_include & w1_include & mask, other=0.0)
        v10 = tl.load(in3_ptr + in3_base + h1_cl * W3 + w0_cl,
                      mask=h1_include & w0_include & mask, other=0.0)
        v11 = tl.load(in3_ptr + in3_base + h1_cl * W3 + w1_cl,
                      mask=h1_include & w1_include & mask, other=0.0)

        pool_max = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

        out_offset = b * C_out * HW2 + c * HW2 + w_offsets
        tl.store(out_ptr + out_offset, pool_max, mask=mask)

    else:
        # ---- relu + scale + bias write (cat's second half, c >= C2) ----
        x2_base = b * C2 * HW2 + c * HW2 + w_offsets
        x2_val  = tl.load(in2_ptr + x2_base, mask=mask, other=0.0)

        in1_val = tl.load(in1_ptr)   # scale [1], always at index 0
        in0_val = tl.load(in0_ptr)   # bias  [1], always at index 0

        x2_f32 = x2_val.to(tl.float32)
        x2_out = tl.maximum(x2_f32, 0.0) * in1_val.to(tl.float32) + in0_val.to(tl.float32)

        out_offset = b * C_out * HW2 + c * HW2 + w_offsets
        tl.store(out_ptr + out_offset, x2_out.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_kernel(in_0, in_1, in_2, in_3):
    B   = in_2.shape[0]
    C2  = in_2.shape[1]   # channels for in2/in3
    H2  = in_2.shape[2]
    W2  = in_2.shape[3]
    H3  = in_3.shape[2]
    W3  = in_3.shape[3]
    HW2 = H2 * W2
    C_out = 2 * C2        # cat along dim=1

    out = torch.empty((B, C_out, H2, W2), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (triton.cdiv(HW2, meta['BLOCK_W']), B * C_out)

    _fused_relu_scale_bias_maxpool_cat[grid](
        in_0, in_1, in_2, in_3, out,
        H2, W2, H3, W3,
        C_out,
        BLOCK_W=128,
    )

    return (out,)


def replacement_func():
    return fused_kernel