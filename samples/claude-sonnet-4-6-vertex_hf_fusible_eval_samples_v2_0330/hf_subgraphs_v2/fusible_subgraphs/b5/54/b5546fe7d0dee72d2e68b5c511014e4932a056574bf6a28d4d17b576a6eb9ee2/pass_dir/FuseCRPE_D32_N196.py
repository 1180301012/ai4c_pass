import torch
import triton
import triton.language as tl

# D=32, N=196 (14x14), scale=0.1767766952966369, output=(1,197,256)

@triton.jit
def fused_crpe_kernel_D32_N196(
    in2_ptr, in3_ptr, conv_ptr,
    in4_ptr, in6_ptr,
    out_ptr,
    C2, C3, C_conv,
    scale,
    N: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    head = tl.program_id(1)

    d_offs = tl.arange(0, BLOCK_D)
    mask_d = d_offs < D

    in4_vals = tl.load(in4_ptr + head * (N + 1) * D + t * D + d_offs, mask=mask_d, other=0.0)
    scaled_in4 = scale * in4_vals

    valid = (t > 0)
    spatial = tl.where(valid, t - 1, 0)
    h = spatial // W
    w = spatial % W
    c_offs = head * D + d_offs

    in2_vals = tl.load(in2_ptr + c_offs * N + h * W + w,
                       mask=(c_offs < C2) & mask_d & valid, other=0.0)
    in3_c = c_offs - C2
    in3_vals = tl.load(in3_ptr + in3_c * N + h * W + w,
                       mask=(c_offs >= C2) & (c_offs < C2 + C3) & mask_d & valid, other=0.0)
    conv_c = c_offs - C2 - C3
    conv_vals = tl.load(conv_ptr + conv_c * N + h * W + w,
                        mask=(c_offs >= C2 + C3) & mask_d & valid, other=0.0)

    cat_vals = tl.where(c_offs < C2, in2_vals, tl.where(c_offs < C2 + C3, in3_vals, conv_vals))

    in6_vals = tl.load(in6_ptr + head * N * D + spatial * D + d_offs, mask=mask_d & valid, other=0.0)

    out_vals = scaled_in4 + tl.where(valid, in6_vals * cat_vals,
                                      tl.zeros([BLOCK_D], dtype=in6_vals.dtype))

    tl.store(out_ptr + t * 8 * D + head * D + d_offs, out_vals, mask=mask_d)


@torch.fx.wrap
def fused_crpe_D32_N196(in_2, in_3, conv_out, in_4, in_6):
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    C_conv = conv_out.shape[1]
    H, W = in_2.shape[2], in_2.shape[3]
    N = H * W
    D = 32
    SCALE = 0.1767766952966369

    out = torch.empty(1, N + 1, 8 * D, dtype=in_4.dtype, device=in_4.device)
    grid = (N + 1, 8)
    fused_crpe_kernel_D32_N196[grid](
        in_2, in_3, conv_out, in_4, in_6, out,
        C2, C3, C_conv, SCALE,
        N=N, D=D, W=W, BLOCK_D=32,
    )
    return out


def pattern(in_2, in_3, conv_out, in_4, in_6):
    tmp_3 = torch.cat([in_2, in_3, conv_out], dim=1)
    tmp_4 = tmp_3.reshape(1, 8, 32, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = 0.1767766952966369 * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 256)
    return tmp_11


def replacement_args(in_2, in_3, conv_out, in_4, in_6):
    return (in_2, in_3, conv_out, in_4, in_6)


def replacement_func():
    return fused_crpe_D32_N196