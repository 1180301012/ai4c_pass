import torch
import triton
import triton.language as tl


@triton.jit
def fused_gelu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Upcast to float32 for precision (important for fp16/bf16)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    gelu_x = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * inv_sqrt2))

    # Fused multiply, cast back to original dtype
    out = (gelu_x * y_f32).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


def _pick_config(n_elements: int):
    """Pick BLOCK_SIZE and num_warps targeting ~1024 GPU blocks.

    - Target 1024 blocks across 56 A30 SMs → ~18 blocks/SM → multiple waves at
      high concurrency, giving ~36+ warps/SM for good erf latency hiding.
    - Use 2 elements per thread for 32-bit (fp16) or 64-bit (fp32) coalesced loads.
      Formula: num_warps = BLOCK_SIZE / (32 threads/warp × 2 elems/thread)
                         = BLOCK_SIZE / 64
    """
    raw = max(1, n_elements // 1024)
    p2 = 1 << (raw - 1).bit_length()           # round up to next power-of-2
    block_size = min(4096, max(128, p2))
    num_warps = max(1, min(16, block_size // 64))  # 2 elements per thread
    return block_size, num_warps


@torch.fx.wrap
def fused_gelu_mul_dropout(in_0, in_1):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    BLOCK_SIZE, num_warps = _pick_config(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_gelu_mul_kernel[grid](
        in_0, in_1, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
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
    return fused_gelu_mul_dropout