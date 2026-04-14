"""
Shared Triton kernels and dispatch function used by both
FuseRollLayerNormAdd_32x32_768 and FuseRollLayerNormAdd_64x64_384 passes.
Both passes import and return the SAME _dispatch_fused_roll_ln_add object
so that replacement_func_limit is never triggered.

Performance notes (A30, NVIDIA):
  * No autotune – avoids repeated JIT overhead during the benchmark warmup window.
  * num_warps=8 → 256 threads/block; for BLOCK_C=1024 each thread handles 4 fp16
    elements which gives a 64-bit vectorised load (LDG.64) per thread.
  * fp32 accumulation for layer-norm accuracy.
"""
import torch
import triton
import triton.language as tl


# ================================================================== #
#  Kernel for C=768  (H=32, W=32, N=1024)
#  768 = 3×256: three unmasked 256-element chunks.
#  num_warps=4 → 16 blocks/SM → 16×108=1728 concurrent → single wave
#  for N=1024 blocks on A30.
# ================================================================== #

@triton.jit
def _roll_ln_add_768_kernel(
    in3_ptr,   # contiguous [1,32,32,768]  (flat: [1024,768])
    in2_ptr,   # residual   [1,1024,768]
    in1_ptr,   # LN weight  [768]
    in0_ptr,   # LN bias    [768]
    out_ptr,   # output     [1,1024,768]
    IS_FP16: tl.constexpr,
):
    CHUNK: tl.constexpr = 256   # 768 = 3 × 256

    row_idx = tl.program_id(0)
    i = row_idx // 32
    j = row_idx % 32

    # roll(shifts=(4,4)): out[i,j] = in3[(i-4)%32,(j-4)%32]
    src_i = (i - 4 + 32) % 32
    src_j = (j - 4 + 32) % 32
    src_k  = src_i * 32 + src_j

    c0 = tl.arange(0, CHUNK)      # 0..255
    c1 = c0 + 256                  # 256..511
    c2 = c0 + 512                  # 512..767

    # ---- load source (3 unmasked chunks) ----
    base3 = in3_ptr + src_k * 768
    x0 = tl.load(base3 + c0).to(tl.float32)
    x1 = tl.load(base3 + c1).to(tl.float32)
    x2 = tl.load(base3 + c2).to(tl.float32)

    # ---- layer-norm mean ----
    mean = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0) + tl.sum(x2, axis=0)) / 768.0

    # ---- variance ----
    d0 = x0 - mean;  d1 = x1 - mean;  d2 = x2 - mean
    var  = (tl.sum(d0*d0, axis=0) + tl.sum(d1*d1, axis=0) + tl.sum(d2*d2, axis=0)) / 768.0
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # ---- normalize + scale + bias + residual, store ----
    base_out = out_ptr + row_idx * 768
    base_res = in2_ptr + row_idx * 768

    w0 = tl.load(in1_ptr + c0).to(tl.float32)
    b0 = tl.load(in0_ptr + c0).to(tl.float32)
    r0 = tl.load(base_res  + c0).to(tl.float32)
    y0 = d0 * rstd * w0 + b0 + r0

    w1 = tl.load(in1_ptr + c1).to(tl.float32)
    b1 = tl.load(in0_ptr + c1).to(tl.float32)
    r1 = tl.load(base_res  + c1).to(tl.float32)
    y1 = d1 * rstd * w1 + b1 + r1

    w2 = tl.load(in1_ptr + c2).to(tl.float32)
    b2 = tl.load(in0_ptr + c2).to(tl.float32)
    r2 = tl.load(base_res  + c2).to(tl.float32)
    y2 = d2 * rstd * w2 + b2 + r2

    if IS_FP16:
        tl.store(base_out + c0, y0.to(tl.float16))
        tl.store(base_out + c1, y1.to(tl.float16))
        tl.store(base_out + c2, y2.to(tl.float16))
    else:
        tl.store(base_out + c0, y0.to(tl.bfloat16))
        tl.store(base_out + c1, y1.to(tl.bfloat16))
        tl.store(base_out + c2, y2.to(tl.bfloat16))


# ================================================================== #
#  Kernel for C=384  (H=64, W=64, N=4096)
#  384 = 3×128: three unmasked 128-element chunks.
#  num_warps=2 → 32 blocks/SM → 32×108=3456 concurrent → ~2 waves
#  for N=4096 blocks (minimum achievable at this N).
# ================================================================== #

@triton.jit
def _roll_ln_add_384_kernel(
    in3_ptr,   # contiguous [1,64,64,384]  (flat: [4096,384])
    in2_ptr,   # residual   [1,4096,384]
    in1_ptr,   # LN weight  [384]
    in0_ptr,   # LN bias    [384]
    out_ptr,   # output     [1,4096,384]
    IS_FP16: tl.constexpr,
):
    CHUNK: tl.constexpr = 128   # 384 = 3 × 128

    row_idx = tl.program_id(0)
    i = row_idx // 64
    j = row_idx % 64

    # roll(shifts=(4,4)): out[i,j] = in3[(i-4)%64,(j-4)%64]
    src_i = (i - 4 + 64) % 64
    src_j = (j - 4 + 64) % 64
    src_k  = src_i * 64 + src_j

    c0 = tl.arange(0, CHUNK)      # 0..127
    c1 = c0 + 128                  # 128..255
    c2 = c0 + 256                  # 256..383

    base3 = in3_ptr + src_k * 384
    x0 = tl.load(base3 + c0).to(tl.float32)
    x1 = tl.load(base3 + c1).to(tl.float32)
    x2 = tl.load(base3 + c2).to(tl.float32)

    mean = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0) + tl.sum(x2, axis=0)) / 384.0

    d0 = x0 - mean;  d1 = x1 - mean;  d2 = x2 - mean
    var  = (tl.sum(d0*d0, axis=0) + tl.sum(d1*d1, axis=0) + tl.sum(d2*d2, axis=0)) / 384.0
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    base_out = out_ptr + row_idx * 384
    base_res = in2_ptr + row_idx * 384

    w0 = tl.load(in1_ptr + c0).to(tl.float32)
    b0 = tl.load(in0_ptr + c0).to(tl.float32)
    r0 = tl.load(base_res  + c0).to(tl.float32)
    y0 = d0 * rstd * w0 + b0 + r0

    w1 = tl.load(in1_ptr + c1).to(tl.float32)
    b1 = tl.load(in0_ptr + c1).to(tl.float32)
    r1 = tl.load(base_res  + c1).to(tl.float32)
    y1 = d1 * rstd * w1 + b1 + r1

    w2 = tl.load(in1_ptr + c2).to(tl.float32)
    b2 = tl.load(in0_ptr + c2).to(tl.float32)
    r2 = tl.load(base_res  + c2).to(tl.float32)
    y2 = d2 * rstd * w2 + b2 + r2

    if IS_FP16:
        tl.store(base_out + c0, y0.to(tl.float16))
        tl.store(base_out + c1, y1.to(tl.float16))
        tl.store(base_out + c2, y2.to(tl.float16))
    else:
        tl.store(base_out + c0, y0.to(tl.bfloat16))
        tl.store(base_out + c1, y1.to(tl.bfloat16))
        tl.store(base_out + c2, y2.to(tl.bfloat16))


# ================================================================== #
#  Shared dispatch wrapper — same object returned by both pass files
# ================================================================== #

@torch.fx.wrap
def _dispatch_fused_roll_ln_add(in_0, in_1, in_2, in_3, route):
    """
    in_0  : LN bias   [C]
    in_1  : LN weight [C]
    in_2  : residual  [1, N, C]
    in_3  : contiguous source tensor  (flat layout = [1, H, W, C])
    route : "768" or "384"
    """
    out     = torch.empty_like(in_2)
    is_fp16 = (in_2.dtype == torch.float16)

    if route == "768":
        # num_warps=4 → 16 blocks/SM → single wave for N=1024
        _roll_ln_add_768_kernel[(1024,)](
            in_3, in_2, in_1, in_0, out,
            IS_FP16=is_fp16,
            num_warps=4,
        )
    elif route == "384":
        # num_warps=2 → 32 blocks/SM → ~2 waves for N=4096
        _roll_ln_add_384_kernel[(4096,)](
            in_3, in_2, in_1, in_0, out,
            IS_FP16=is_fp16,
            num_warps=2,
        )

    return out