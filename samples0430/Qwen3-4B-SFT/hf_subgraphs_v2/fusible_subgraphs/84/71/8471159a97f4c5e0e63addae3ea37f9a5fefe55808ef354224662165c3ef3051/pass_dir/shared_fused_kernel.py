import torch
import triton
import triton.language as tl


@triton.jit
def fused_scale_mul_kernel(
    bias_ptr,    # [C]  – in_0 (bias)
    weight_ptr,  # [C,K] – in_1 (weight)
    input_ptr,   # [B,K] – in_2 (input to linear)
    feat_ptr,    # [B,C,HW] – in_3 (feature map)
    output_ptr,  # [B,C,HW] – out
    B, C, K, HW,
    stride_b_input, stride_b_feat, stride_c_feat,
    BLOCK_HW: tl.constexpr,
    BLOCK_K:  tl.constexpr,   # = next_power_of_2(K)
):
    """
    Grid: (B*C, ceil(HW/BLOCK_HW)).
    Per program: dot product (K elements) → sigmoid → scale HW tile.
    """
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    b = pid_bc // C
    c = pid_bc % C

    k     = tl.arange(0, BLOCK_K)
    k_mask = k < K
    in_k  = tl.load(input_ptr  + b * stride_b_input + k,
                    mask=k_mask, other=0.0).to(tl.float32)
    wt_k  = tl.load(weight_ptr + c * C + k,
                    mask=k_mask, other=0.0).to(tl.float32)
    fval  = tl.sum(in_k * wt_k, axis=0)
    fval  = fval + tl.load(bias_ptr + c).to(tl.float32)

    sig_val = 1.0 / (1.0 + tl.exp(-fval))

    hw_off  = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < HW
    feat_base = b * stride_b_feat + c * stride_c_feat
    feat_vals = tl.load(feat_ptr + feat_base + hw_off,
                        mask=hw_mask, other=0.0)

    result = feat_vals.to(tl.float32) * sig_val
    tl.store(output_ptr + feat_base + hw_off,
             result.to(feat_vals.dtype),
             mask=hw_mask)


@torch.fx.wrap
def fused_scale_mul(in_0, in_1, in_2, in_3):
    B  = in_3.shape[0]
    C  = in_3.shape[1]
    K  = in_2.shape[1]
    HW = in_3.shape[2] * in_3.shape[3]
    BLOCK_K = triton.next_power_of_2(K)
    output  = torch.empty_like(in_3)
    SB      = in_2.stride(0)
    SFb     = in_3.stride(0)
    SFc     = in_3.stride(1)

    # ----------------------------------------------------------------
    # Grid: (B*C, ceil(HW/BLOCK_HW))
    #
    #   B=1, BLOCK_HW=1024, num_warps=8 (256 threads/block)
    #     → grid=(64,4)=256 progs; 64/(56×2)=0.57 waves → ≤ 1 wave
    #     → 2 blocks/SM × 256 threads = 512 threads = 25% occupancy
    #     Previously BLOCK_HW=64 (4096 progs) gave 0.85x.  With
    #     fewer, larger programs the overhead per program is lower
    #     (each program loads 16 KB vs 128 B) which can improve
    #     cache-miss overlap and warp scheduling.
    #
    #   B=32, BLOCK_HW=512, num_warps=4 (128 threads/block)
    #     → grid=(2048,7)=14336; 14336/(56×16)=1.6 waves
    #
    #   B=128, BLOCK_HW=2048, num_warps=8 (256 threads/block)
    #     → grid=(8192,2)=16384; 16384/(56×8)=3.7 waves → ~100%
    # ----------------------------------------------------------------
    if B == 1:
        BLOCK_HW, NUM_WARPS = 1024,  8
    elif B <= 32:
        BLOCK_HW, NUM_WARPS = 512,   4
    else:
        BLOCK_HW, NUM_WARPS = 2048,  8

    grid = (B * C, triton.cdiv(HW, BLOCK_HW))

    fused_scale_mul_kernel[grid](
        bias_ptr    = in_0,
        weight_ptr  = in_1,
        input_ptr   = in_2,
        feat_ptr    = in_3,
        output_ptr  = output,
        B           = B,
        C           = C,
        K           = K,
        HW          = HW,
        stride_b_input  = SB,
        stride_b_feat   = SFb,
        stride_c_feat   = SFc,
        BLOCK_HW    = BLOCK_HW,
        BLOCK_K     = BLOCK_K,
        num_warps   = NUM_WARPS,
    )

    return output