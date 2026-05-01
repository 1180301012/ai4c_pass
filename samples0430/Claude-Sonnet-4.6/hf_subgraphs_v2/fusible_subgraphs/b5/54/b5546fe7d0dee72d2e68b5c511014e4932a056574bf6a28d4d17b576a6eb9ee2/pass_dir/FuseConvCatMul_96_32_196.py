"""
Fused kernel for: conv2d(depthwise) + cat + reshape + transpose(-1,-2) + mul
Specific to: groups=96, C_head=32, N=196 (coat_lite_tiny/medium stage-3 with 14x14 input)

C2=64, C3=96, C_conv=96, H=W=14
in_2:       [1, 64,  14, 14]
in_3:       [1, 96,  14, 14]
in_5 (conv_input): [1, 96, 14, 14]
in_1 (weight):     [96, 1, 7, 7]
in_0 (bias):       [96]
in_6:       [1, 8, 196, 32]
output:     [1, 8, 196, 32]
"""
import torch
import triton
import triton.language as tl


def pattern(in_5, in_1, in_0, in_2, in_3, in_6):
    conv_out = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), 96)
    tmp_3    = torch.cat([in_2, in_3, conv_out], dim=1)
    tmp_4    = tmp_3.reshape(1, 8, 32, 196)
    tmp_5    = tmp_4.transpose(-1, -2)
    tmp_6    = in_6 * tmp_5
    return tmp_6


def replacement_args(in_5, in_1, in_0, in_2, in_3, in_6):
    return (in_5, in_1, in_0, in_2, in_3, in_6)


# ---------------------------------------------------------------------------
# Triton kernel – fused depthwise-conv + direct-read + transpose-mul
# No intermediate cat tensor is created.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_C': 64}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _fused_conv_cat_mul_96_32_196(
    in5_ptr, in1_ptr, in0_ptr,
    in2_ptr, in3_ptr,
    in6_ptr, out_ptr,
    N,          # 196
    C_head,     # 32
    C2,         # 64
    C3,         # 96
    W,          # 14 (spatial width)
    # in5/in2/in3 strides (each is [1, C, H, W]):
    #   ptr[c, h, w] = ptr + c * N + h * W + w  (for contiguous)
    # in6 strides [1, 8, N, C_head]:
    in6_s1, in6_s2, in6_s3,   # stride(1)=N*C_head, stride(2)=C_head, stride(3)=1
    # out same layout as in6
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Grid: (8, ceil(N/BLOCK_N), ceil(C_head/BLOCK_C))
    Head h selects which source to read from.
    """
    pid_h  = tl.program_id(0)   # head ∈ [0,7]
    pid_nb = tl.program_id(1)   # N-block
    pid_cb = tl.program_id(2)   # C-block

    n_off = pid_nb * BLOCK_N
    c_off = pid_cb * BLOCK_C

    n_range = n_off + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    c_range = c_off + tl.arange(0, BLOCK_C)   # [BLOCK_C]
    n_mask  = n_range < N
    c_mask  = c_range < C_head

    # spatial decomposition
    h_out = n_range // W     # [BLOCK_N]
    w_out = n_range % W      # [BLOCK_N]

    # ---- load in6 tile [BLOCK_N, BLOCK_C] (rows = spatial, cols = channel) ----
    in6_base = pid_h * in6_s1
    in6_off  = n_range[:, None] * in6_s2 + c_range[None, :]
    in6_tile = tl.load(in6_ptr + in6_base + in6_off,
                       mask=n_mask[:, None] & c_mask[None, :], other=0.0)

    # ---- compute cat tile [BLOCK_C, BLOCK_N] then transpose ----
    # channel = pid_h * C_head + c_range   shape [BLOCK_C]
    channel = pid_h * C_head + c_range    # [BLOCK_C]

    # Head-based source selection flags (scalar bool, broadcast via mask)
    h_is_in2  = pid_h < 2                          # heads 0-1
    h_is_in3  = (pid_h >= 2) & (pid_h < 5)        # heads 2-4
    h_is_conv = pid_h >= 5                          # heads 5-7

    # ---- in2 tile [BLOCK_C, BLOCK_N]: mask out for non-in2 heads ----
    # in2[channel, n]  at  ptr + channel * N + n
    in2_off  = channel[:, None] * N + n_range[None, :]   # [BLOCK_C, BLOCK_N]
    in2_mask = h_is_in2 & (c_mask[:, None] & n_mask[None, :])
    in2_tile = tl.load(in2_ptr + in2_off, mask=in2_mask, other=0.0)

    # ---- in3 tile [BLOCK_C, BLOCK_N]: mask out for non-in3 heads ----
    # in3[channel-C2, n]
    in3_chan  = channel - C2                              # [BLOCK_C]
    in3_off   = in3_chan[:, None] * N + n_range[None, :]  # [BLOCK_C, BLOCK_N]
    in3_mask  = h_is_in3 & (c_mask[:, None] & n_mask[None, :])
    in3_tile  = tl.load(in3_ptr + in3_off, mask=in3_mask, other=0.0)

    # ---- conv tile [BLOCK_C, BLOCK_N]: mask out for non-conv heads ----
    conv_chan = channel - C2 - C3        # [BLOCK_C] channel index in in5

    # Load bias (safe: h_is_conv guard prevents invalid pointer deref)
    bias_vals = tl.load(in0_ptr + conv_chan,
                        mask=h_is_conv & c_mask, other=0.0)
    conv_tile = tl.zeros([BLOCK_C, BLOCK_N], dtype=tl.float32)
    conv_tile += bias_vals[:, None].to(tl.float32)

    # 7x7 depthwise conv (loop is unrolled by Triton)
    H_SPATIAL = N // W   # = 14 for W=14, N=196
    for kh in range(7):
        for kw in range(7):
            h_in = h_out + kh - 3   # [BLOCK_N]
            w_in = w_out + kw - 3   # [BLOCK_N]
            valid_n = (h_in >= 0) & (h_in < H_SPATIAL) & \
                      (w_in >= 0) & (w_in < W)
            in5_off = conv_chan[:, None] * N + h_in[None, :] * W + w_in[None, :]
            in5_mask = h_is_conv & (c_mask[:, None] & (valid_n & n_mask)[None, :])
            in5_val = tl.load(in5_ptr + in5_off, mask=in5_mask, other=0.0)
            w_off = conv_chan * 49 + kh * 7 + kw   # [BLOCK_C]
            w_val = tl.load(in1_ptr + w_off,
                            mask=h_is_conv & c_mask, other=0.0)
            conv_tile += in5_val.to(tl.float32) * w_val[:, None].to(tl.float32)

    conv_tile_out = conv_tile.to(in6_tile.dtype)

    # Select source and transpose to [BLOCK_N, BLOCK_C]
    cat_tile = tl.where(h_is_in2, in2_tile,
                        tl.where(h_is_in3, in3_tile, conv_tile_out))
    cat_T = tl.trans(cat_tile)   # [BLOCK_N, BLOCK_C]

    # Element-wise multiply and store
    result   = in6_tile * cat_T
    out_base = pid_h * in6_s1
    tl.store(out_ptr + out_base + in6_off, result,
             mask=n_mask[:, None] & c_mask[None, :])


@torch.fx.wrap
def fused_conv_cat_mul_96_32_196(in_5, in_1, in_0, in_2, in_3, in_6):
    H_      = 8
    C_head_ = 32
    N_      = 196
    W_      = 14
    C2_     = 64
    C3_     = 96

    out = torch.empty_like(in_6)
    grid = lambda meta: (H_,
                         triton.cdiv(N_,      meta['BLOCK_N']),
                         triton.cdiv(C_head_, meta['BLOCK_C']))
    _fused_conv_cat_mul_96_32_196[grid](
        in_5, in_1, in_0, in_2, in_3, in_6, out,
        N_, C_head_, C2_, C3_, W_,
        in_6.stride(1), in_6.stride(2), in_6.stride(3),
    )
    return out


def replacement_func():
    return fused_conv_cat_mul_96_32_196