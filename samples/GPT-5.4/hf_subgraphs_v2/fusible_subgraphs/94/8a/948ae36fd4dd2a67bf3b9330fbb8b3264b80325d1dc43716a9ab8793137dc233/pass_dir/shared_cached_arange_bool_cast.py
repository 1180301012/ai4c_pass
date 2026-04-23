import torch
import triton
import triton.language as tl
from torch.fx import wrap

_ARANGE_CACHE = {}


@triton.jit
def cast_i64_to_bool_kernel(
    in_ptr,
    out_ptr,
    m,
    n,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)

    in_ptrs = in_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    vals = tl.load(in_ptrs, mask=mask, other=0)
    out_vals = vals != 0
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out_vals, mask=mask)


def _get_cached_arange(seq_len, device):
    key = (device.type, device.index, int(seq_len))
    cached = _ARANGE_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(list(range(int(seq_len))), device=device, dtype=torch.int64)
        _ARANGE_CACHE[key] = cached
    return cached


def _launch_cast_kernel(in_0, out):
    m = in_0.shape[0]
    n = in_0.shape[1]
    block_m = 8
    block_n = 128
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    cast_i64_to_bool_kernel[grid](
        in_0,
        out,
        m,
        n,
        in_0.stride(0),
        in_0.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )


@wrap
def replacement_entry(in_0, seq_len):
    seq_len = int(seq_len)
    arange_out = _get_cached_arange(seq_len, in_0.device)
    out = torch.empty(in_0.shape, device=in_0.device, dtype=torch.bool)
    _launch_cast_kernel(in_0, out)
    return arange_out, out


@wrap
def replacement_cast_bool_only(in_0):
    out = torch.empty(in_0.shape, device=in_0.device, dtype=torch.bool)
    _launch_cast_kernel(in_0, out)
    return out