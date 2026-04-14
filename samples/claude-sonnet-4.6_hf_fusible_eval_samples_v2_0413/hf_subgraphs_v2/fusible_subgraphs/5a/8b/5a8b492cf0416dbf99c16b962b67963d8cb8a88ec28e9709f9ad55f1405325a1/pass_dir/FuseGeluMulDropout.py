import torch
import triton
import triton.language as tl


# ── float32 kernel ────────────────────────────────────────────────────────────
@triton.jit
def fused_gelu_mul_fp32_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in0_ptr + offsets, mask=mask, cache_modifier=".cg")
    y = tl.load(in1_ptr + offsets, mask=mask, cache_modifier=".cg")

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_out = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(out_ptr + offsets, gelu_out * y, mask=mask,
             eviction_policy="evict_first")


# ── float16 / bfloat16 kernel: upcast to fp32 for erf ────────────────────────
@triton.jit
def fused_gelu_mul_lowp_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in0_ptr + offsets, mask=mask)
    y = tl.load(in1_ptr + offsets, mask=mask)

    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)

    gelu_out = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))
    result = (gelu_out * y_f32).to(x.dtype)

    tl.store(out_ptr + offsets, result, mask=mask,
             eviction_policy="evict_first")


@torch.fx.wrap
def fused_gelu_mul(in0, in1):
    n_elements = in0.numel()
    out = torch.empty_like(in0)

    if in0.dtype == torch.float32:
        # float32: BLOCK_SIZE=1024 gives ~56% SM occupancy (better than 2048's 25%)
        # with .cg for streaming loads — empirically best for fp32
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_gelu_mul_fp32_kernel[grid](
            in0, in1, out, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    else:
        # bf16/fp16: BLOCK_SIZE=2048 enables 128-bit vectorised loads
        # (16 bytes/thread × 32 threads/warp = 512 bytes/warp = 4 cache lines)
        # No .cg: default cache path is better for this size
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_gelu_mul_lowp_kernel[grid](
            in0, in1, out, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

    return out


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_gelu_mul