import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_mask_kernel(
    in5_ptr,
    out_ptr,
    N,
    NEG_INF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: in_5.to(float32) -> 1.0 - x -> to_bool -> masked_fill(-inf)

    For each element x (int64):
      val  = 1.0 - float(x)
      bool = (val != 0.0)
      out  = -inf if bool else val    (i.e. if val==0 → 0.0, else → -inf)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(in5_ptr + offsets, mask=mask, other=0).to(tl.float32)
    val = 1.0 - x
    # masked_fill fills where the bool is True (i.e. where val != 0)
    result = tl.where(val != 0.0, NEG_INF, val)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_attention_mask(in_5):
    """Fused: to_float32 → 1 - x → to_bool → masked_fill(-inf)"""
    N = in_5.numel()
    out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_attention_mask_kernel[grid](
        in5_ptr=in_5,
        out_ptr=out,
        N=N,
        NEG_INF=-3.4028234663852886e+38,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


# ── Pattern & Replacement ──────────────────────────────────────────────────────

def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5):
    return (in_5,)


def replacement_func():
    return triton_attention_mask