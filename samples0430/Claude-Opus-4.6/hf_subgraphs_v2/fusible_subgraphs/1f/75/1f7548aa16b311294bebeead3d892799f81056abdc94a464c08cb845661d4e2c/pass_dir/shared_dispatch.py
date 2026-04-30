import torch
import triton
import triton.language as tl


@triton.jit
def causal_mask_kernel(
    in_0_ptr, in_2_ptr, out_ptr,
    B, S, total_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems

    j = offsets % S
    rem = offsets // S
    i = rem % S
    b = rem // S

    cache_pos = tl.load(in_2_ptr + i, mask=mask, other=0)
    causal = j <= cache_pos

    attn_val = tl.load(in_0_ptr + b * S + j, mask=mask, other=0)
    attn = attn_val != 0

    result = causal & attn
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def cast_to_float32_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float32), mask=mask)


@triton.jit
def cast_int64_to_float32_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x.to(tl.float32), mask=mask)


def _compute_causal_mask(in_0, in_2):
    B = in_0.shape[0]
    S = in_0.shape[1]
    total_elems = B * S * S
    out_mask = torch.empty((B, 1, S, S), dtype=torch.bool, device=in_0.device)
    BLOCK_SIZE = 1024
    grid = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    causal_mask_kernel[grid](in_0, in_2, out_mask, B, S, total_elems, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out_mask


def _compute_inv_freq(in_1):
    D = in_1.shape[0]
    out = torch.empty((1, D, 1), dtype=torch.float32, device=in_1.device)
    BLOCK_SIZE = max(triton.next_power_of_2(D), 16)
    cast_to_float32_kernel[(1,)](in_1, out, D, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _compute_pos_ids(in_3):
    B = in_3.shape[0]
    S = in_3.shape[1]
    N = B * S
    out = torch.empty((B, 1, S), dtype=torch.float32, device=in_3.device)
    BLOCK_SIZE = min(max(triton.next_power_of_2(N), 16), 1024)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    cast_int64_to_float32_kernel[grid](in_3, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


@torch.fx.wrap
def dispatch_fn(arg0, arg1, route):
    if route == "causal_mask":
        return _compute_causal_mask(arg0, arg1)
    elif route == "inv_freq":
        return _compute_inv_freq(arg0)
    elif route == "pos_ids":
        return _compute_pos_ids(arg0)