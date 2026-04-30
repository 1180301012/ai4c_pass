import torch
import triton
import triton.language as tl


@triton.jit
def _rope_key_k(out_ptr, in_4_ptr, in_0_ptr, H, S, BLOCK_D: tl.constexpr):
    row_idx = tl.program_id(0)
    h = row_idx // (S + 1)
    s = row_idx % (S + 1)
    d = tl.arange(0, BLOCK_D)
    base = h * (S + 1) * BLOCK_D + s * BLOCK_D
    x = tl.load(in_4_ptr + base + d)
    rope_s = (s + S - 1) % S
    cos = tl.load(in_0_ptr + rope_s * (2 * BLOCK_D) + BLOCK_D + d)
    sin = tl.load(in_0_ptr + rope_s * (2 * BLOCK_D) + d)
    partner_d = d ^ 1
    x_partner = tl.load(in_4_ptr + base + partner_d)
    rotated = tl.where(d % 2 == 0, -x_partner, x_partner)
    rope_val = x * cos + rotated * sin
    val = tl.where(s > 0, rope_val, x)
    tl.store(out_ptr + base + d, val)


@triton.jit
def _rope_query_k(out_ptr, in_2_ptr, in_3_ptr, in_1_ptr, in_5_ptr, H, S, BLOCK_D: tl.constexpr):
    row_idx = tl.program_id(0)
    h = row_idx // (S + 1)
    s = row_idx % (S + 1)
    d = tl.arange(0, BLOCK_D)
    out_base = h * (S + 1) * BLOCK_D + s * BLOCK_D
    cls_val = tl.load(in_2_ptr + h * BLOCK_D + d)
    rope_s = (s + S - 1) % S
    in_3_base = h * S * BLOCK_D + rope_s * BLOCK_D
    x = tl.load(in_3_ptr + in_3_base + d)
    cos = tl.load(in_1_ptr + rope_s * BLOCK_D + d)
    sin = tl.load(in_5_ptr + rope_s * BLOCK_D + d)
    x_partner = tl.load(in_3_ptr + in_3_base + (d ^ 1))
    rotated = tl.where(d % 2 == 0, -x_partner, x_partner)
    rope_val = x * cos + rotated * sin
    val = tl.where(s > 0, rope_val, cls_val)
    tl.store(out_ptr + out_base + d, val)


def fused_rope_dispatch(*args):
    route = args[-1]
    if route.startswith("q"):
        in_1, in_2, in_3, in_5, in_6, _route = args
        H = in_3.shape[1]
        S = in_3.shape[2]
        D = 64
        out = torch.empty(1, H, S + 1, D, dtype=in_6.dtype, device=in_3.device)
        grid = (H * (S + 1),)
        _rope_query_k[grid](out, in_2, in_3, in_1, in_5, H, S, BLOCK_D=D, num_warps=2, num_stages=2)
        return out
    else:
        in_0, in_4, in_6, _route = args
        H = in_4.shape[1]
        S = in_4.shape[2] - 1
        D = 64
        out = torch.empty(1, H, S + 1, D, dtype=in_6.dtype, device=in_4.device)
        grid = (H * (S + 1),)
        _rope_key_k[grid](out, in_4, in_0, H, S, BLOCK_D=D, num_warps=2, num_stages=2)
        return out