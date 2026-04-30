import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch._C._nn.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch._C._nn.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim=2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _load_embedding_row(ids_row_ptr, weight_ptr, s, S, offs_d):
    valid_s = (s >= 0) & (s < S)
    safe_s = tl.where(valid_s, s, 0)
    idx = tl.load(ids_row_ptr + safe_s).to(tl.int32)
    base = idx * 128
    return tl.load(weight_ptr + base + offs_d, mask=valid_s, other=0.0)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 2}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 8}, num_warps=4, num_stages=2),
    ],
    key=["S"],
)
@triton.jit
def _embedding_shift_pad_cat_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    B,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    num_s_blocks = tl.cdiv(S, BLOCK_S)
    b = pid // num_s_blocks
    sb = pid % num_s_blocks
    s0 = sb * BLOCK_S

    offs_d = tl.arange(0, 128)
    ids_row_ptr = ids_ptr + b * S

    row_prev = _load_embedding_row(ids_row_ptr, weight_ptr, s0 - 1, S, offs_d)
    row_center = _load_embedding_row(ids_row_ptr, weight_ptr, s0, S, offs_d)
    row_next = _load_embedding_row(ids_row_ptr, weight_ptr, s0 + 1, S, offs_d)

    for i in range(BLOCK_S):
        s = s0 + i
        valid_out = s < S
        out_base = (b * S + s) * 384
        tl.store(out_ptr + out_base + offs_d, row_next, mask=valid_out)
        tl.store(out_ptr + out_base + 128 + offs_d, row_center, mask=valid_out)
        tl.store(out_ptr + out_base + 256 + offs_d, row_prev, mask=valid_out)
        row_prev = row_center
        row_center = row_next
        row_next = _load_embedding_row(ids_row_ptr, weight_ptr, s + 2, S, offs_d)


@torch.fx.wrap
def fused_embedding_shift_pad_cat_seqdim1(in_0, in_1):
    B, S = in_0.shape
    out = torch.empty((B, S, 384), device=in_1.device, dtype=in_1.dtype)

    grid = lambda meta: (B * triton.cdiv(S, meta["BLOCK_S"]),)
    _embedding_shift_pad_cat_kernel[grid](
        in_0,
        in_1,
        out,
        B,
        S,
    )
    return out


def replacement_func():
    return fused_embedding_shift_pad_cat_seqdim1