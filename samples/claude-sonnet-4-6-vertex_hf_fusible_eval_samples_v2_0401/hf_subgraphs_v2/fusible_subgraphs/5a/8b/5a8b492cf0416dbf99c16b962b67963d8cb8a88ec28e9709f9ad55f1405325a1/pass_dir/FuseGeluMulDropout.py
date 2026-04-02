import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – must mirror model.py exactly
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Kernel A: with bounds mask (general, safe for any n_elements)
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_mul_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    # Compute GELU in fp32 for accuracy, then multiply in original dtype.
    # This saves registers (no yf = y.to(float32) needed) and matches
    # PyTorch's fp16/bf16 multiply semantics exactly.
    gelu_f32 = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    out = gelu_f32.to(x.dtype) * y
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Kernel B: mask-free fast path (n_elements % BLOCK_SIZE == 0)
# No bounds check → cleaner vectorisation, lower instruction count.
# ---------------------------------------------------------------------------
@triton.jit
def _gelu_mul_kernel_exact(
    x_ptr, y_ptr, out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    xf = x.to(tl.float32)
    # Compute GELU in fp32 for accuracy, then multiply in original dtype.
    # Saves ~12 registers/thread vs upcasting y to fp32 separately:
    #   Old: x(4) + xf(8) + y(4) + yf(8) + gelu_f32(8) + out_f32(8) + out(4) = 44
    #   New: x(4) + xf(8) + y(4) + gelu_f32(8) + gelu_dtype(4) + out(4) = 32
    # Fewer regs → more active blocks/SM → higher warp occupancy → better BW.
    gelu_f32 = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    out = gelu_f32.to(x.dtype) * y
    tl.store(out_ptr + offsets, out)


# ---------------------------------------------------------------------------
# Block-size / warp heuristics
#
#   Tile size           Threads/block     Elements/thread    Load width
#   BLOCK_SIZE=256      128  (4 wrps)     2                  32-bit
#   BLOCK_SIZE=1024     128  (4 wrps)     8                  128-bit ✓
#   BLOCK_SIZE=4096     256  (8 wrps)     16                 256-bit ✓
#   BLOCK_SIZE=4096     512  (16 wrps)    8                  128-bit, full SM occupancy
#
# Key rule: for memory-bound element-wise kernels, we want:
#   1. ≥128-bit vectorised loads (≥8 elements/thread for fp16/bf16)
#   2. Enough blocks to fill all 56 SMs (A30)
#   3. Full SM occupancy (2048 threads/SM) for large tensors
# ---------------------------------------------------------------------------
def _pick_block_cfg(n: int, dtype_bytes: int):
    """
    Choose (BLOCK_SIZE, NUM_WARPS) tuned per dtype.

    BLOCK_SIZE=1024, NW=4 (128 threads):
      - fp32:  8 elems/thread = 32B = 256-bit load  → 4096 blk for 4M, 73 waves
      - fp16/bf16: 8 elems/thread = 16B = 128-bit load → same block count/waves
    This gives ~10 active blocks/SM → 62–65% warp occupancy on A30.

    BLOCK_SIZE=4096, NW=16 (512 threads) for large bf16 (16M):
      - 8 fp16/thread = 128-bit; 4096 blocks; excellent occupancy for L2-bypass region.
    """
    if dtype_bytes == 4:  # float32 – 4-byte elements
        if n <= 65_536:
            return 512,  4   # small: 88+ blocks; 4 f32/thread = 128-bit
        else:
            return 1024, 4   # medium/large: 4096 blocks for 4M; 8 f32/thread = 256-bit
    else:  # float16 / bfloat16 – 2-byte elements
        if n <= 8_388_608:   # ≤8M (22K / 128K / 4M)
            return 1024, 4   # 4096 blocks for 4M; 8 fp16/thread = 128-bit
        else:                # >8M (16M bfloat16)
            return 2048, 8   # 8192 blocks → 146 waves; 8 fp16/thread = 128-bit loads


@torch.fx.wrap
def gelu_mul_dropout_fused(x, y):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE, NUM_WARPS = _pick_block_cfg(n_elements, x.element_size())
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    if n_elements % BLOCK_SIZE == 0:
        # Mask-free path – all test tensors have last-dim 2048, which is a
        # multiple of every BLOCK_SIZE we use, so this branch always fires.
        _gelu_mul_kernel_exact[(num_blocks,)](
            x_ptr=x, y_ptr=y, out_ptr=out,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
    else:
        _gelu_mul_kernel[(num_blocks,)](
            x_ptr=x, y_ptr=y, out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )

    return out


# ---------------------------------------------------------------------------
# Replacement entry-point (zero-argument factory returning a callable)
# ---------------------------------------------------------------------------
def replacement_func():
    return gelu_mul_dropout_fused