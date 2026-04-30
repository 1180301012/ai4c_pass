import torch
import triton
import triton.language as tl


@triton.jit
def fused_identity_mean_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    hw,
    inv_hw,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * hw
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in tl.range(0, hw, BLOCK_SIZE):
        idx = start + offs
        mask = idx < hw
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        tl.store(out_ptr + base + idx, x, mask=mask)
        acc += x.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) * inv_hw
    tl.store(mean_ptr + row, mean_val)


@triton.jit
def fused_binary_mean_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    mean_ptr,
    hw,
    inv_hw,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * hw
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in tl.range(0, hw, BLOCK_SIZE):
        idx = start + offs
        mask = idx < hw
        a = tl.load(a_ptr + base + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + base + idx, mask=mask, other=0.0)
        out = a + b
        tl.store(out_ptr + base + idx, out, mask=mask)
        acc += out.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) * inv_hw
    tl.store(mean_ptr + row, mean_val)


@triton.jit
def fused_ternary_mean_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    mean_ptr,
    hw,
    inv_hw,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * hw
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in tl.range(0, hw, BLOCK_SIZE):
        idx = start + offs
        mask = idx < hw
        a = tl.load(a_ptr + base + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + base + idx, mask=mask, other=0.0)
        c = tl.load(c_ptr + base + idx, mask=mask, other=0.0)
        tmp = a + b
        out = tmp + c
        tl.store(out_ptr + base + idx, out, mask=mask)
        acc += out.to(tl.float32)

    mean_val = tl.sum(acc, axis=0) * inv_hw
    tl.store(mean_ptr + row, mean_val)


def _select_launch_config(hw):
    if hw <= 64:
        return 64, 2
    if hw <= 256:
        return 128, 4
    if hw <= 1024:
        return 256, 4
    return 256, 8


def _alloc_outputs(x):
    out = torch.empty_like(x)
    mean = torch.empty((x.shape[0], x.shape[1], 1, 1), device=x.device, dtype=x.dtype)
    return out, mean


def _run_identity(x):
    hw = x.shape[-2] * x.shape[-1]
    nrows = x.numel() // hw
    out, mean = _alloc_outputs(x)
    block_size, num_warps = _select_launch_config(hw)
    fused_identity_mean_kernel[(nrows,)](
        x,
        out,
        mean,
        hw,
        1.0 / float(hw),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out, mean


def _run_binary(a, b):
    hw = a.shape[-2] * a.shape[-1]
    nrows = a.numel() // hw
    out, mean = _alloc_outputs(a)
    block_size, num_warps = _select_launch_config(hw)
    fused_binary_mean_kernel[(nrows,)](
        a,
        b,
        out,
        mean,
        hw,
        1.0 / float(hw),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out, mean


def _run_ternary(a, b, c):
    hw = a.shape[-2] * a.shape[-1]
    nrows = a.numel() // hw
    out, mean = _alloc_outputs(a)
    block_size, num_warps = _select_launch_config(hw)
    fused_ternary_mean_kernel[(nrows,)](
        a,
        b,
        c,
        out,
        mean,
        hw,
        1.0 / float(hw),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out, mean


@torch.fx.wrap
def dispatch_fused_add_mean(*args):
    route = args[-1]
    if route == "identity":
        return _run_identity(args[0])
    if route == "binary":
        return _run_binary(args[0], args[1])
    if route == "ternary":
        return _run_ternary(args[0], args[1], args[2])
    raise RuntimeError(f"Unknown fused-add-mean route: {route}")