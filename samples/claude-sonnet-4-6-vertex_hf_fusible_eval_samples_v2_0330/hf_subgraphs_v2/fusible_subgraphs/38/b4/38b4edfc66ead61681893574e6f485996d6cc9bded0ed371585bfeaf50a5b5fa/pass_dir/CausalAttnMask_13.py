import torch
import triton
import triton.language as tl
from torch import device

# Force torch.tensor to trace as call_function (not constant-folded to get_attr)
torch.fx.wrap(torch.tensor)

# ── Triton kernel ────────────────────────────────────────────────────────────
# Output: float32 tensor of shape (1, 1, 13, 13)
# Logic per element at linear offset k  (row = k // 13, col = k % 13):
#   out = 0.0  if  col <= row  AND  in_0[0, col] != 0
#   out = NEG_INF  otherwise

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
    ],
    key=[],
)
@triton.jit
def _causal_attn_mask_kernel_13(
    in_ptr,   # [1, 13] int64 attention mask
    out_ptr,  # [1, 1, 13, 13] float32
    BLOCK_SIZE: tl.constexpr,
):
    N = 13
    NEG_INF = -3.4028234663852886e+38

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N * N

    row = offsets // N   # i
    col = offsets % N    # j

    # causal: j <= i
    causal_ok = col <= row

    # attention mask: in_0[0, j] != 0
    attn_val = tl.load(in_ptr + col, mask=valid, other=0)
    attn_ok = attn_val != 0

    out = tl.where(causal_ok & attn_ok, 0.0, NEG_INF)
    tl.store(out_ptr + offsets, out, mask=valid)


@torch.fx.wrap
def _fused_causal_attn_mask_13(in_0):
    N = 13
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    n_elem = N * N   # 169
    grid = lambda meta: ((n_elem + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _causal_attn_mask_kernel_13[grid](in_0, out)
    return out


# ── Pattern (mirrors model.py exactly) ───────────────────────────────────────
def pattern(in_0):
    tmp_1 = torch.full((13, 13), -3.4028234663852886e+38, device=device(type='cuda', index=0))
    tmp_2 = torch.arange(13, device=device(type='cuda', index=0))
    tmp_3 = tmp_2 + 1
    tmp_4 = tmp_3.view(13, 1)
    tmp_5 = tmp_2 < tmp_4
    tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
    tmp_7 = tmp_6.to(torch.float32)
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, 13, 13)
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 13, 13)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_13 = torch.tensor(1.0, dtype=torch.float32)
    tmp_14 = tmp_13 - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _fused_causal_attn_mask_13