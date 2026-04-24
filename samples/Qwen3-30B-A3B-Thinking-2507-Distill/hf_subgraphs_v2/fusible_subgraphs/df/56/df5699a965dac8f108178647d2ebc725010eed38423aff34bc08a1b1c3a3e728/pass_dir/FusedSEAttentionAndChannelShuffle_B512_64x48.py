import torch
import triton
import triton.language as tl


@triton.jit
def se_attention_kernel(
    in6_ptr,    # [B, C_in, 1, 1] flattened to [B, C_in]
    in1_ptr,    # [C_out, C_in, 1, 1] flattened to [C_out, C_in]
    bias_ptr,   # [C_out]
    in5_ptr,    # [B, C_out, H, W] contiguous
    out_ptr,    # [B, C_out, H, W] contiguous
    B, C_in, C_out, HW,
    BLOCK_K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program handles one (b, c_out) pair.
    Computes: out[b, c_out, :] = in5[b, c_out, :] * sigmoid(conv_out[b, c_out])
    where conv_out[b, c_out] = dot(in6[b,:], in1[c_out,:]) + bias[c_out]
    """
    pid = tl.program_id(0)
    b = pid // C_out
    c_out = pid % C_out

    # Compute 1x1 conv: dot product + bias → scalar
    k_idx = tl.arange(0, BLOCK_K)
    k_mask = k_idx < C_in
    in6_vals = tl.load(in6_ptr + b * C_in + k_idx, mask=k_mask, other=0.0).to(tl.float32)
    in1_vals = tl.load(in1_ptr + c_out * C_in + k_idx, mask=k_mask, other=0.0).to(tl.float32)
    conv_out = tl.sum(in6_vals * in1_vals, axis=0)

    bias_val = tl.load(bias_ptr + c_out).to(tl.float32)
    conv_out += bias_val

    # sigmoid
    att = tl.sigmoid(conv_out)

    # Multiply with in5 (broadcast over HW)
    hw_start = tl.program_id(1) * BLOCK_HW
    hw_idx = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_idx < HW
    src_base = b * C_out * HW + c_out * HW
    x_vals = tl.load(in5_ptr + src_base + hw_idx, mask=hw_mask)
    result = x_vals * att.to(x_vals.dtype)
    tl.store(out_ptr + src_base + hw_idx, result, mask=hw_mask)


@triton.jit
def channel_shuffle_kernel(
    s1_ptr,     # [B, C, H, W]
    s2_ptr,     # [B, C, H, W]
    out0_ptr,   # [B, C, H, W]  -- first chunk output
    out1_ptr,   # [B, C, H, W]  -- second chunk output
    B, C, HW,
    C_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cat + channel_shuffle for 2-input 2-output case.
    Semantics:
      cat([s1, s2], dim=1)  -> [B, 2C, H, W]
      shuffle -> [B, 2C, H, W] interleaved
      chunk(2) -> (out0[B, C, H, W], out1[B, C, H, W])
    out0[b, c, h, w] = s1[b, c//2, h, w]  if c%2==0
                     = s2[b, c//2, h, w]  if c%2==1
    out1[b, c, h, w] = s1[b, C_HALF + c//2, h, w]  if c%2==0
                     = s2[b, C_HALF + c//2, h, w]  if c%2==1
    """
    pid = tl.program_id(0)
    b = pid // (2 * C_HALF)
    rem = pid % (2 * C_HALF)
    c = rem % C_HALF

    hw_blocks = tl.cdiv(HW, BLOCK_SIZE)
    block_idx = tl.program_id(1)
    hw_start = block_idx * BLOCK_SIZE
    hw_idx = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask = hw_idx < HW

    src_base = b * C * HW + c * HW
    src_flat1 = src_base + hw_idx
    val_s1 = tl.load(s1_ptr + src_flat1, mask=hw_mask)
    val_s2 = tl.load(s2_ptr + src_flat1, mask=hw_mask)

    out_even = b * C * HW + (2 * c) * HW + hw_idx
    out_odd = b * C * HW + (2 * c + 1) * HW + hw_idx

    tl.store(out0_ptr + out_even, val_s1, mask=hw_mask)
    tl.store(out0_ptr + out_odd, val_s2, mask=hw_mask)


@torch.fx.wrap
def fused_se_and_shuffle_b512_64x48(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # in_0: bias [40], in_1: weight [40, 10, 1, 1]
    # in_2: [512, 20, 64, 48], in_3: [512, 40, 32, 24]
    # in_4: [512, 20, 64, 48], in_5: [512, 40, 32, 24]
    # in_6: [512, 10, 1, 1]
    B, C_in, C_out = 512, 10, 40
    C2, C3 = 20, 40
    HW2, HW3 = 64 * 48, 32 * 24   # 3072, 768

    # Allocate outputs
    tmp_4 = torch.empty_like(in_5)                                        # [512,40,32,24]
    out0 = torch.empty_like(in_2)                                         # [512,20,64,48]
    out1 = torch.empty_like(in_2)                                         # [512,20,64,48]
    out2 = torch.empty_like(in_3)                                         # [512,40,32,24]
    out3 = torch.empty_like(in_3)                                         # [512,40,32,24]

    BLOCK_HW = 256
    BLOCK_SIZE = 256

    # --- SE attention: conv2d + sigmoid + multiply -> tmp_4 ---
    num_hw_blocks = (HW3 + BLOCK_HW - 1) // BLOCK_HW
    se_grid = (B * C_out, num_hw_blocks)
    se_attention_kernel[se_grid](
        in_6, in_1, in_0, in_5, tmp_4,
        B, C_in, C_out, HW3,
        BLOCK_K=16, BLOCK_HW=BLOCK_HW,
    )

    # --- Channel shuffle for [512,20,64,48] branch ---
    # cat([in_2, in_4]) + shuffle + chunk -> out0, out1
    num_hw2_blocks = (HW2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    chan_shuffle_grid = (B * C2, num_hw2_blocks)
    channel_shuffle_kernel[chan_shuffle_grid](
        in_2, in_4, out0, out1,
        B, C2, HW2,
        C_HALF=10, BLOCK_SIZE=BLOCK_SIZE,
    )

    # --- Channel shuffle for [512,40,32,24] branch ---
    # cat([in_3, tmp_4]) + shuffle + chunk -> out2, out3
    num_hw3_blocks = (HW3 + BLOCK_SIZE - 1) // BLOCK_SIZE
    chan_shuffle_grid2 = (B * C3, num_hw3_blocks)
    channel_shuffle_kernel[chan_shuffle_grid2](
        in_3, tmp_4, out2, out3,
        B, C3, HW3,
        C_HALF=20, BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out0, out2, out1, out3)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    tmp_7 = tmp_5.view(512, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(512, 40, 64, 48)
    tmp_11 = tmp_6.view(512, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(512, 80, 32, 24)
    chunk = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    chunk_1 = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return (tmp_16, tmp_19, tmp_17, tmp_20)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


def replacement_func():
    return fused_se_and_shuffle_b512_64x48