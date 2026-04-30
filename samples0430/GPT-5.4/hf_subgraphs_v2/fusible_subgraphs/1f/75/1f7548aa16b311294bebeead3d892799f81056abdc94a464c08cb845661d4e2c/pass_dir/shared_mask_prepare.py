import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit
def causal_mask_kernel(
    attn_mask_ptr,
    cache_pos_ptr,
    out_ptr,
    B,
    S,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    cache_val = tl.load(cache_pos_ptr + pid_r)
    causal = offs_s <= cache_val

    attn_base = attn_mask_ptr + pid_b * S
    attn_vals = tl.load(attn_base + offs_s, mask=mask_s, other=0)
    attn_bool = attn_vals != 0

    out = causal & attn_bool

    out_base = out_ptr + pid_b * S * S + pid_r * S
    tl.store(out_base + offs_s, out, mask=mask_s)


@torch.fx.wrap
def fused_prepare_outputs(in_0, in_1, in_2, in_3):
    # Output 1: fused bool causal-attention mask, shape [B, 1, S, S]
    B = in_0.shape[0]
    S = in_0.shape[1]
    out0 = torch.empty((B, 1, S, S), device=in_0.device, dtype=torch.bool)

    block_s = _next_power_of_2(S)
    if block_s < 16:
        block_s = 16
    if block_s > 1024:
        block_s = 1024

    causal_mask_kernel[(B, S)](
        in_0,
        in_2,
        out0,
        B,
        S,
        BLOCK_S=block_s,
    )

    # Output 2: preserve graph semantics; this is effectively float(in_1)[None, :, None]
    # The original graph does expand + to(cuda) + float() again, but result shape stays [1, 64, 1].
    out1 = in_1[None, :, None].float()

    # Output 3: float(position_ids[:, None, :])
    out2 = in_3[:, None, :].float()

    return out0, out1, out2


def get_replacement_func():
    return fused_prepare_outputs