import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Fused 3x3-conv(stride=1,pad=1) + max-pool(k=3,s=2,p=1) Triton kernel
#
# Grid: (N * OC,  ceil(H_pool * W_pool / BLOCK_PIXELS))
#
# Conv params:  stride=1 pad=1 dilation=1 kernel=3x3
# Pool params:  kernel=3 stride=2 padding=1 dilation=1
#
# Combined:
#   h_conv = hp*2 + ph - 1  (pool stride=2, pad=1)
#   h_in   = h_conv + kh - 1  (conv stride=1, pad=1)
#          = hp*2 + ph + kh - 2
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_PIXELS': 16}, num_warps=4),
        triton.Config({'BLOCK_PIXELS': 32}, num_warps=4),
        triton.Config({'BLOCK_PIXELS': 64}, num_warps=8),
    ],
    key=['N', 'H_pool', 'W_pool', 'IC'],
)
@triton.jit
def fused_conv3x3s1p1_pool3x3s2p1_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, OC, H_in, W_in, H_pool, W_pool,
    IC:           tl.constexpr,
    BLOCK_PIXELS: tl.constexpr,
    IS_FP16:      tl.constexpr,
    IS_BF16:      tl.constexpr,
):
    nc_idx  = tl.program_id(0)   # n * OC + oc
    pix_blk = tl.program_id(1)

    n  = nc_idx // OC
    oc = nc_idx %  OC

    pix_offs = pix_blk * BLOCK_PIXELS + tl.arange(0, BLOCK_PIXELS)
    pix_mask = pix_offs < H_pool * W_pool

    hp = (pix_offs // W_pool).to(tl.int32)
    wp = (pix_offs %  W_pool).to(tl.int32)

    if IS_FP16:
        neg_inf = tl.full([BLOCK_PIXELS], float('-inf'), dtype=tl.float16)
    elif IS_BF16:
        neg_inf = tl.full([BLOCK_PIXELS], float('-inf'), dtype=tl.bfloat16)
    else:
        neg_inf = tl.full([BLOCK_PIXELS], float('-inf'), dtype=tl.float32)

    max_val = neg_inf

    input_base  = n.to(tl.int64) * IC * H_in * W_in
    weight_base = oc.to(tl.int64) * IC * 9  # IC * 3*3

    # Pool window: stride=2, pad=1, kernel=3
    for ph in range(3):
        for pw in range(3):
            # Combined h_in = hp*2 + ph + kh - 2  (kh in 0..2)
            h_conv = hp * 2 + ph - 1   # conv output position
            w_conv = wp * 2 + pw - 1

            # conv output valid range is [0, H_in) for stride=1, pad=1 with 3x3 kernel
            conv_valid = (pix_mask
                          & (h_conv >= 0) & (h_conv < H_in)
                          & (w_conv >= 0) & (w_conv < W_in))

            conv_val = tl.zeros([BLOCK_PIXELS], dtype=tl.float32)

            # Conv kernel: 3x3, IC channels
            for ic in range(IC):
                ic_in_off = ic.to(tl.int64) * H_in * W_in
                ic_w_off  = ic.to(tl.int64) * 9
                for kh in range(3):
                    for kw in range(3):
                        h_in = h_conv + kh - 1
                        w_in = w_conv + kw - 1
                        valid = (conv_valid
                                 & (h_in >= 0) & (h_in < H_in)
                                 & (w_in >= 0) & (w_in < W_in))
                        in_idx = (input_base + ic_in_off
                                  + h_in.to(tl.int64) * W_in
                                  + w_in.to(tl.int64))
                        in_val = tl.load(input_ptr + in_idx, mask=valid, other=0.0)
                        w_idx  = weight_base + ic_w_off + kh * 3 + kw
                        w_val  = tl.load(weight_ptr + w_idx)
                        conv_val += in_val.to(tl.float32) * w_val.to(tl.float32)

            if IS_FP16:
                max_val = tl.maximum(max_val, conv_val.to(tl.float16))
            elif IS_BF16:
                max_val = tl.maximum(max_val, conv_val.to(tl.bfloat16))
            else:
                max_val = tl.maximum(max_val, conv_val)

    out_base = (n.to(tl.int64) * OC + oc) * H_pool * W_pool
    out_idx  = out_base + hp.to(tl.int64) * W_pool + wp.to(tl.int64)
    tl.store(output_ptr + out_idx, max_val, mask=pix_mask)


@torch.fx.wrap
def fused_conv_maxpool_s1p1(in_0, in_1):
    """in_0 = weight [OC, IC, 3, 3], in_1 = input [N, IC, H, W]."""
    if not in_1.is_cuda:
        in_0 = in_0.cuda()
        in_1 = in_1.cuda()

    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()

    N,  IC, H_in, W_in = in_1.shape
    OC = in_0.shape[0]

    # Conv output size for stride=1, pad=1, kernel=3:  H_conv = H_in
    H_conv = H_in
    W_conv = W_in

    # Pool output size for stride=2, pad=1, kernel=3:
    H_pool = (H_conv - 1) // 2 + 1
    W_pool = (W_conv - 1) // 2 + 1
    NC     = N * OC

    output = torch.empty((N, OC, H_pool, W_pool),
                         dtype=in_1.dtype, device=in_1.device)

    is_fp16 = in_1.dtype == torch.float16
    is_bf16 = in_1.dtype == torch.bfloat16

    def grid(meta):
        return (NC, triton.cdiv(H_pool * W_pool, meta['BLOCK_PIXELS']))

    fused_conv3x3s1p1_pool3x3s2p1_kernel[grid](
        in_1, in_0, output,
        N, OC, H_in, W_in, H_pool, W_pool,
        IC=IC,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    return output


def replacement_func():
    return fused_conv_maxpool_s1p1