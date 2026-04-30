import torch
import triton
import triton.language as tl


_MEAN_CACHE = {}


@triton.jit
def _copy_mean_contig_kernel(
    inp_ptr,
    out_ptr,
    mean_ptr,
    c_in,
    c_out,
    c_offset,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    in_plane = pid_n * c_in + pid_c
    out_plane = pid_n * c_out + pid_c + c_offset

    inp_base = inp_ptr + in_plane * HW
    out_base = out_ptr + out_plane * HW

    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in tl.static_range(0, HW, BLOCK_SIZE):
        idx = start + offs
        mask = idx < HW
        x = tl.load(inp_base + idx, mask=mask, other=0.0)
        tl.store(out_base + idx, x, mask=mask)
        acc += x.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) / HW
    tl.store(mean_ptr + out_plane, mean_val)


@triton.jit
def _copy_mean_strided_kernel(
    inp_ptr,
    out_ptr,
    mean_ptr,
    inp_sn,
    inp_sc,
    inp_sh,
    inp_sw,
    out_sn,
    out_sc,
    out_sh,
    out_sw,
    mean_sn,
    mean_sc,
    c_offset,
    W,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in tl.static_range(0, HW, BLOCK_SIZE):
        idx = start + offs
        mask = idx < HW
        h = idx // W
        w = idx % W
        inp_ptrs = inp_ptr + pid_n * inp_sn + pid_c * inp_sc + h * inp_sh + w * inp_sw
        out_ptrs = out_ptr + pid_n * out_sn + (pid_c + c_offset) * out_sc + h * out_sh + w * out_sw
        x = tl.load(inp_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, x, mask=mask)
        acc += x.to(tl.float32)

    mean_ptrs = mean_ptr + pid_n * mean_sn + (pid_c + c_offset) * mean_sc
    mean_val = tl.sum(acc, axis=0) / HW
    tl.store(mean_ptrs, mean_val)


@triton.jit
def _mean_contig_kernel(
    inp_ptr,
    out_ptr,
    c_in,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    in_plane = pid_n * c_in + pid_c
    inp_base = inp_ptr + in_plane * HW
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.static_range(0, HW, BLOCK_SIZE):
        idx = start + offs
        mask = idx < HW
        x = tl.load(inp_base + idx, mask=mask, other=0.0)
        acc += x.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) / HW
    out_idx = pid_n * c_in + pid_c
    tl.store(out_ptr + out_idx, mean_val)


@triton.jit
def _mean_strided_kernel(
    inp_ptr,
    out_ptr,
    inp_sn,
    inp_sc,
    inp_sh,
    inp_sw,
    out_sn,
    out_sc,
    W,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in tl.static_range(0, HW, BLOCK_SIZE):
        idx = start + offs
        mask = idx < HW
        h = idx // W
        w = idx % W
        inp_ptrs = inp_ptr + pid_n * inp_sn + pid_c * inp_sc + h * inp_sh + w * inp_sw
        x = tl.load(inp_ptrs, mask=mask, other=0.0)
        acc += x.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) / HW
    out_ptrs = out_ptr + pid_n * out_sn + pid_c * out_sc
    tl.store(out_ptrs, mean_val)


def _pick_launch_config(hw):
    if hw <= 64:
        return 64, 2
    if hw <= 256:
        return 128, 4
    if hw <= 1024:
        return 256, 4
    return 256, 8


def _cat_impl(in_0, in_1):
    n, c0, h, w = in_0.shape
    c1 = in_1.shape[1]
    total_c = c0 + c1
    out = torch.empty((n, total_c, h, w), device=in_0.device, dtype=in_0.dtype)
    mean = torch.empty((n, total_c, 1, 1), device=in_0.device, dtype=in_0.dtype)
    hw = h * w
    block_size, num_warps = _pick_launch_config(hw)

    if in_0.is_contiguous() and in_1.is_contiguous() and out.is_contiguous() and mean.is_contiguous():
        _copy_mean_contig_kernel[(c0, n)](
            in_0,
            out,
            mean,
            c0,
            total_c,
            0,
            HW=hw,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        _copy_mean_contig_kernel[(c1, n)](
            in_1,
            out,
            mean,
            c1,
            total_c,
            c0,
            HW=hw,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        _copy_mean_strided_kernel[(c0, n)](
            in_0,
            out,
            mean,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            in_0.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            mean.stride(0),
            mean.stride(1),
            0,
            w,
            HW=hw,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        _copy_mean_strided_kernel[(c1, n)](
            in_1,
            out,
            mean,
            in_1.stride(0),
            in_1.stride(1),
            in_1.stride(2),
            in_1.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            mean.stride(0),
            mean.stride(1),
            c0,
            w,
            HW=hw,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

    _MEAN_CACHE[out.data_ptr()] = mean
    return out


def _mean_impl(x):
    cached = _MEAN_CACHE.pop(x.data_ptr(), None)
    if cached is not None:
        return cached

    n, c, h, w = x.shape
    out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)
    hw = h * w
    block_size, num_warps = _pick_launch_config(hw)

    if x.is_contiguous() and out.is_contiguous():
        _mean_contig_kernel[(c, n)](
            x,
            out,
            c,
            HW=hw,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        _mean_strided_kernel[(c, n)](
            x,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(0),
            out.stride(1),
            w,
            HW=hw,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    return out


@torch.fx.wrap
def dispatch_cat_or_mean(x, y, route):
    if route == 'cat':
        return _cat_impl(x, y)
    return _mean_impl(x)


def replacement_func():
    return dispatch_cat_or_mean